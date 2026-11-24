"""MODO EN VIVO · cámara + YOLO + MLP + semáforos en tiempo real.

Cada fotograma se toma según el reloj real (como una cámara), YOLOv8n detecta
los vehículos y el semáforo avanza segundo a segundo. Al comenzar cada ciclo,
la MLP decide los segundos de verde con lo que la cámara ve en ese momento
(mediana de los últimos segundos de conteo).

Este módulo no depende de Streamlit: contiene la fuente de video y el
controlador del semáforo, por eso se puede probar por separado.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

from analisis import Decision, LaneCount, decide
from traffic import ALL_RED

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class LiveSource:
    """Una 'cámara': devuelve el fotograma que corresponde al segundo t (el video se repite)."""

    def __init__(self, path: Path, max_width: int = 960):
        import cv2

        self.path, self.max_width = Path(path), max_width
        self.is_image = self.path.suffix.lower() in IMAGE_SUFFIXES
        self.capture, self.position, self.last = None, -1, None
        if self.is_image:
            from PIL import Image, ImageOps

            with Image.open(self.path) as image:
                self.last = self._resize(np.asarray(ImageOps.exif_transpose(image).convert('RGB')))
            self.fps, self.frames = 1.0, 1
        else:
            self.capture = cv2.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                raise ValueError(f'No se pudo abrir el video {self.path.name}.')
            self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
            if not 1 <= self.fps <= 240:
                self.fps = 25.0
            self.frames = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    @property
    def duration(self) -> float:
        return 0.0 if self.is_image else self.frames / self.fps

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if frame.shape[1] > self.max_width:
            height = round(frame.shape[0] * self.max_width / frame.shape[1])
            frame = cv2.resize(frame, (self.max_width, height), interpolation=cv2.INTER_AREA)
        return frame

    def frame_at(self, seconds: float) -> tuple[float, np.ndarray]:
        """(segundo del video, fotograma RGB). Salta fotogramas con grab() para ir a tiempo real."""
        import cv2

        if self.is_image:
            return 0.0, self.last
        target = int(seconds * self.fps) % self.frames
        if target < self.position:  # el video terminó: vuelve a empezar
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.position = -1
        while self.position < target - 1:
            if not self.capture.grab():
                break
            self.position += 1
        ok, frame = self.capture.read()
        if ok:
            self.position += 1
            self.last = self._resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif self.last is None:
            raise ValueError(f'No se pudieron leer fotogramas de {self.path.name}.')
        return self.position / self.fps, self.last

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


class CountHistory:
    """Conteos recientes de una avenida; la MLP usa su mediana (robusta a errores momentáneos)."""

    def __init__(self, seconds: float = 3.0):
        self.seconds = seconds
        self.items: deque[tuple[float, int, int, dict]] = deque()

    def add(self, now: float, total: int, heavy: int, by_class: dict) -> None:
        self.items.append((now, total, heavy, by_class))
        while self.items and now - self.items[0][0] > self.seconds:
            self.items.popleft()

    def current(self) -> LaneCount:
        if not self.items:
            return LaneCount(0, 0)
        total = round(median(item[1] for item in self.items))
        heavy = min(total, round(median(item[2] for item in self.items)))
        return LaneCount(total, heavy, dict(self.items[-1][3]))


@dataclass
class SignalState:
    """Lo que muestran los dos semáforos en un instante."""
    a: str
    b: str
    remaining_a: float
    remaining_b: float
    decision: Decision | None
    new_decision: bool = False


@dataclass
class LiveSignal:
    """Semáforo de dos avenidas que pide una decisión a la MLP al empezar cada ciclo."""
    controller: object
    use_saturation: bool = True
    cycle_start: float | None = None
    phases: list[dict] = field(default_factory=list)
    decision: Decision | None = None
    decisions: list[Decision] = field(default_factory=list)

    def update(self, t: float, lane_a: LaneCount, lane_b: LaneCount) -> SignalState:
        """t = segundos desde que se pulsó Iniciar."""
        if t < ALL_RED:  # despeje inicial: todo rojo
            return SignalState('Rojo', 'Rojo', ALL_RED - t, ALL_RED - t, self.decision)
        new = False
        if self.cycle_start is None or t >= self.cycle_start + self._duration():
            start = ALL_RED if self.cycle_start is None else self.cycle_start + self._duration()
            # Si la app se retrasó varios ciclos, salta al ciclo actual.
            while self.cycle_start is not None and t >= start + self._duration() and self._duration():
                start += self._duration()
            # >>> Nueva decisión de la MLP con lo que la cámara ve ahora <<<
            self.decision = decide(self.controller, lane_a, lane_b, self.use_saturation,
                                   label=f'ciclo {len(self.decisions) + 1}', start=t)
            self.decisions.append(self.decision)
            self.phases = self.decision.phases if self.decision.cycle_seconds else [{'a': 'Rojo', 'b': 'Rojo', 's': 1}]
            self.cycle_start, new = start, True
        position = t - self.cycle_start
        for index, phase in enumerate(self.phases):
            if position < phase['s']:
                remaining = phase['s'] - position
                return SignalState(phase['a'], phase['b'], self._lane_left('a', index, remaining),
                                   self._lane_left('b', index, remaining), self.decision, new)
            position -= phase['s']
        last = self.phases[-1]
        return SignalState(last['a'], last['b'], 0.0, 0.0, self.decision, new)

    def _duration(self) -> float:
        return sum(phase['s'] for phase in self.phases)

    def _lane_left(self, lane: str, index: int, remaining: float) -> float:
        """Segundos hasta que esa avenida cambie de color dentro del ciclo."""
        color = self.phases[index][lane]
        for phase in self.phases[index + 1:]:
            if phase[lane] != color:
                break
            remaining += phase['s']
        return remaining


LAMP = {'Rojo': 'red', 'Amarillo': 'amber', 'Verde': 'green'}
WORD = {'Rojo': 'ALTO', 'Amarillo': 'PRECAUCIÓN', 'Verde': 'SIGA'}


def signal_html(name: str, color: str, remaining: float, green: int | None, vehicles: int) -> str:
    """Semáforo dibujado con HTML/CSS (los estilos están en style.css)."""
    lamps = ''.join(f'<i class="tl-lamp {c}{" on" if LAMP[color] == c else ""}"></i>' for c in ('red', 'amber', 'green'))
    green_text = f'Verde: <b>{green} s</b>' if green is not None else 'Esperando decisión'
    return (f'<div class="tl-card tl-{LAMP[color]}"><div class="tl-housing">{lamps}</div>'
            f'<div class="tl-info"><div class="tl-name">{name}</div>'
            f'<div class="tl-state">{color} · {WORD[color]}</div>'
            f'<div class="tl-timer">{max(0, int(remaining + 0.999))}<small> s</small></div>'
            f'<div class="tl-meta">{green_text}<br>{vehicles} vehículos ahora</div></div></div>')
