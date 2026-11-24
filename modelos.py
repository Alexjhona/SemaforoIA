"""Carga única de los dos modelos entrenados y su estado para la interfaz.

* best.pt ................. YOLOv8n (detección de vehículos)
* controlador_tiempos.pt .. MLP (segundos de verde por avenida)

Los pesos originales solo se leen; nunca se modifican ni se vuelven a entrenar.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / 'models' / 'best.pt'
MLP_WEIGHTS = ROOT / 'models' / 'controlador_tiempos.pt'
TRAINING_METRICS = ROOT / '.metricas_colab.json'


@dataclass
class ModelStatus:
    name: str
    path: Path
    loaded: bool = False
    error: str = ''
    load_seconds: float = 0.0
    details: dict[str, str] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1e6 if self.path.is_file() else 0.0


def configure_cpu() -> None:
    """Usa la CPU con un número moderado de hilos para no congelar la laptop."""
    import torch

    threads = max(1, min(8, (os.cpu_count() or 2) - 1))
    torch.set_num_threads(threads)


def load_yolo(path: Path = YOLO_WEIGHTS):
    """Devuelve (detector o None, estado)."""
    status = ModelStatus('YOLOv8n · detector', path)
    if not path.is_file():
        status.error = 'No se encontró models/best.pt'
        return None, status
    try:
        from detector import VehicleDetector

        configure_cpu()
        start = perf_counter()
        detector = VehicleDetector(path)
        detector.warm_up()
        status.load_seconds = perf_counter() - start
        status.loaded = True
        status.details = {
            'Clases': ', '.join(sorted(set(detector.labels.values()))),
            'Dispositivo': 'CPU',
            'Entrada': '640 × 640 px',
        }
        return detector, status
    except Exception as error:  # se informa en pantalla en lugar de detener la app
        status.error = f'{type(error).__name__}: {error}'
        return None, status


def load_mlp(path: Path = MLP_WEIGHTS):
    """Devuelve (controlador o None, estado)."""
    status = ModelStatus('MLP · tiempos de verde', path)
    if not path.is_file():
        status.error = 'No se encontró models/controlador_tiempos.pt'
        return None, status
    try:
        import torch
        from controller import TimeController

        start = perf_counter()
        controller = TimeController(path)
        status.load_seconds = perf_counter() - start
        saved = torch.load(path, map_location='cpu', weights_only=True)
        status.loaded = True
        status.details = {
            'Arquitectura': '6 → 64 → 32 → 2 (ReLU)',
            'Mejor época': str(saved.get('best_epoch', '—')),
            'Rango entrenado': f'A ≤ {controller.count_limits[0]} · B ≤ {controller.count_limits[1]} vehículos',
        }
        return controller, status
    except Exception as error:
        status.error = f'{type(error).__name__}: {error}'
        return None, status


def training_metrics() -> dict:
    """Métricas reales obtenidas en Colab (archivo generado durante el entrenamiento)."""
    try:
        return json.loads(TRAINING_METRICS.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
