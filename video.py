"""Lectura eficiente de videos MP4 para la CPU.

En lugar de analizar los 25-30 fotogramas de cada segundo, se toma una muestra
(por ejemplo 2 fotogramas por segundo). Los fotogramas que no se analizan se
saltan con `grab()`, que no los decodifica por completo, lo que ahorra tiempo y
memoria en una laptop sin GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @property
    def seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def _open(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError('No se pudo abrir el video. Usa un archivo MP4 (H.264) válido.')
    return capture


def video_info(path: Path) -> VideoInfo:
    import cv2

    capture = _open(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if not 1 <= fps <= 240:
            fps = 25.0
        return VideoInfo(fps, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                         int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def sample_frames(path: Path, samples_per_second: float, max_seconds: float,
                  max_width: int = 960) -> Iterator[tuple[float, np.ndarray]]:
    """Devuelve (segundo del video, fotograma RGB) para los fotogramas muestreados.

    * samples_per_second: frecuencia de procesamiento elegida en la interfaz.
    * max_seconds: solo se analiza el inicio del video (videos cortos).
    * max_width: los fotogramas grandes se reducen antes de pasar a YOLO, que de
      todas formas trabaja a 640 px; así se usa menos memoria.
    """
    import cv2

    info = video_info(path)
    step = max(1, round(info.fps / max(samples_per_second, 0.1)))
    last = min(info.frames, int(max_seconds * info.fps)) if info.frames > 0 else int(max_seconds * info.fps)
    capture = _open(path)
    try:
        index = 0
        while index < last:
            if index % step == 0:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] > max_width:
                    frame = cv2.resize(frame, (max_width, round(frame.shape[0] * max_width / frame.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                yield index / info.fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif not capture.grab():
                break
            index += 1
    finally:
        capture.release()


def count_samples(path: Path, samples_per_second: float, max_seconds: float) -> int:
    """Número de fotogramas que se analizarán (para la barra de progreso)."""
    info = video_info(path)
    step = max(1, round(info.fps / max(samples_per_second, 0.1)))
    last = min(info.frames, int(max_seconds * info.fps)) if info.frames > 0 else int(max_seconds * info.fps)
    return max(1, -(-last // step))
