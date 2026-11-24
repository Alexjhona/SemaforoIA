"""ETAPA 2 · Del conteo de YOLO a la decisión de la MLP.

Flujo:
  conteos de YOLO por avenida ──► entradas de la MLP ──► segundos de verde
  ──► plan de fases seguro (traffic.build_cycle) ──► simulación de semáforos

Para videos, los conteos de cada fotograma muestreado se agrupan en "ventanas
de decisión" (por ejemplo, cada 4 s de video). En cada ventana se usa la mediana
de los conteos, que es robusta frente a detecciones momentáneas, y la MLP
calcula un nuevo par de tiempos de verde.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from statistics import median

from traffic import ALL_RED, build_cycle


@dataclass
class LaneCount:
    """Conteo de una avenida en una foto, un fotograma o una ventana de video."""
    total: int
    heavy: int
    by_class: dict[str, int] = field(default_factory=dict)


@dataclass
class FrameSample:
    """Fotograma de video ya analizado por YOLO (se guarda en JPEG pequeño)."""
    time: float
    count: LaneCount
    jpeg: bytes


@dataclass
class Decision:
    """Decisión del controlador neuronal para un ciclo semafórico."""
    label: str
    start: float
    end: float
    detected: tuple[LaneCount, LaneCount]
    mlp_input: tuple[int, int, int, int]        # (vehículos A, vehículos B, pesados A, pesados B)
    saturated: tuple[bool, bool]
    greens: tuple[int, int]                     # segundos de verde predichos por la MLP
    phases: list[dict]

    @property
    def cycle_seconds(self) -> int:
        return sum(phase['s'] for phase in self.phases)


def saturate(total: int, heavy: int, limit: int) -> tuple[int, int, bool]:
    """Limita el conteo al máximo observado al entrenar la MLP.

    La MLP solo vio hasta `limit` vehículos por vía (escenarios de SUMO). Si YOLO
    cuenta más, la avenida se considera saturada: se usa el máximo conocido y se
    conserva la proporción de vehículos pesados.
    """
    if total <= limit:
        return total, heavy, False
    return limit, min(limit, round(heavy * limit / total)), True


def decide(controller, lane_a: LaneCount, lane_b: LaneCount, use_saturation: bool = True,
           label: str = 'Fotografía', start: float = 0.0, end: float = 0.0) -> Decision:
    """Pasa los conteos a la MLP y construye el plan de fases seguro."""
    limit_a, limit_b = controller.count_limits
    if use_saturation:
        count_a, heavy_a, sat_a = saturate(lane_a.total, lane_a.heavy, int(limit_a))
        count_b, heavy_b, sat_b = saturate(lane_b.total, lane_b.heavy, int(limit_b))
    else:
        (count_a, heavy_a, sat_a), (count_b, heavy_b, sat_b) = \
            (lane_a.total, lane_a.heavy, False), (lane_b.total, lane_b.heavy, False)
    # >>> Aquí la red neuronal MLP predice los segundos de verde de cada avenida <<<
    greens = tuple(controller.predict(count_a, count_b, heavy_a, heavy_b))
    phases = [{'a': p.a, 'b': p.b, 's': p.seconds} for p in build_cycle(greens)]
    return Decision(label, start, end, (lane_a, lane_b), (count_a, count_b, heavy_a, heavy_b),
                    (sat_a, sat_b), greens, phases)


def window_count(samples: list[FrameSample], start: float, end: float, duration: float) -> LaneCount:
    """Mediana de los conteos de una avenida dentro de una ventana de tiempo.

    Si un video es más corto que el otro, se repite en bucle (igual que en la
    simulación) para que ambas avenidas tengan datos en cada ventana.
    """
    if not samples:
        return LaneCount(0, 0)
    offset = start % duration if duration > 0 else start
    inside = [s for s in samples if offset <= s.time < offset + (end - start)]
    if not inside:  # ventana sin muestras: se usa la muestra más cercana
        inside = [min(samples, key=lambda s: abs(s.time - offset))]
    classes = {name for s in inside for name in s.count.by_class}
    by_class = {name: round(median(s.count.by_class.get(name, 0) for s in inside)) for name in classes}
    total = round(median(s.count.total for s in inside))
    heavy = min(total, round(median(s.count.heavy for s in inside)))
    return LaneCount(total, heavy, {k: v for k, v in by_class.items() if v})


def video_decisions(controller, samples_a: list[FrameSample], samples_b: list[FrameSample],
                    duration_a: float, duration_b: float, window: float,
                    use_saturation: bool = True) -> list[Decision]:
    """Una decisión de la MLP por cada ventana de video."""
    duration = max(duration_a, duration_b, window)
    decisions = []
    for index in range(max(1, ceil(duration / window - 1e-9))):
        start, end = index * window, min(duration, (index + 1) * window)
        lane_a = window_count(samples_a, start, end, duration_a)
        lane_b = window_count(samples_b, start, end, duration_b)
        decisions.append(decide(controller, lane_a, lane_b, use_saturation,
                                f'{start:.0f}–{end:.0f} s', start, end))
    return decisions


__all__ = ['ALL_RED', 'Decision', 'FrameSample', 'LaneCount', 'decide', 'saturate',
           'video_decisions', 'window_count']
