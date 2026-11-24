"""Modelo de regresión para los tiempos de luz verde de dos vías."""

from pathlib import Path

import torch
from torch import nn

from traffic import MAX_GREEN, MIN_GREEN, bounded_green


FEATURES = ('vehiculos_a', 'vehiculos_b', 'pesados_a', 'pesados_b', 'demanda_total', 'proporcion_a')


def traffic_features(count_a: int, count_b: int, heavy_a: int, heavy_b: int) -> tuple[float, ...]:
    """Prepara variables que están disponibles tanto en SUMO como en una foto por vía."""
    if not (0 <= heavy_a <= count_a and 0 <= heavy_b <= count_b):
        raise ValueError('El conteo de vehículos pesados debe estar dentro del conteo total de cada vía.')
    total = count_a + count_b
    return count_a, count_b, heavy_a, heavy_b, total, count_a / total if total else 0.5


class GreenTimeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2),
        )

    def forward(self, x):
        return MIN_GREEN + (MAX_GREEN - MIN_GREEN) * torch.sigmoid(self.layers(x))


class TimeController:
    def __init__(self, weights: Path):
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        if (saved['format_version'] != 1 or tuple(saved['features']) != FEATURES
                or tuple(saved['green_limits']) != (MIN_GREEN, MAX_GREEN)):
            raise ValueError('El checkpoint MLP tiene una configuración incompatible.')
        self.model = GreenTimeMLP()
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()
        self.mean, self.scale = saved['mean'], saved['scale']
        self.count_limits = saved['count_limits']

    def predict(self, count_a: int, count_b: int, heavy_a: int, heavy_b: int) -> tuple[int, int]:
        features = traffic_features(count_a, count_b, heavy_a, heavy_b)
        if count_a + count_b == 0:
            return 0, 0
        with torch.inference_mode():
            normalized = (torch.tensor(features, dtype=torch.float32) - self.mean) / self.scale
            green_a, green_b = self.model(normalized).tolist()
        return bounded_green(green_a), bounded_green(green_b)
