"""Plan de fases para dos vías en la simulación académica."""

from dataclasses import dataclass
from math import isfinite


MIN_GREEN, MAX_GREEN = 10, 35
YELLOW, ALL_RED = 3, 2


@dataclass(frozen=True)
class Phase:
    a: str
    b: str
    seconds: int


def demonstration_greens(count_a: int, count_b: int) -> tuple[int, int]:
    """Línea base proporcional, disponible para evaluación y demostración sin pesos."""
    if min(count_a, count_b) < 0:
        raise ValueError('El conteo de vehículos debe ser positivo o cero.')
    total = count_a + count_b
    if total == 0:
        return 0, 0
    green_a = bounded_green(45 * count_a / total)
    return green_a, 45 - green_a


def bounded_green(seconds: float) -> int:
    """Convierte una predicción finita a segundos enteros dentro de los límites."""
    if not isfinite(seconds):
        raise ValueError('El controlador produjo una duración inválida.')
    return min(MAX_GREEN, max(MIN_GREEN, round(seconds)))


def build_cycle(greens: tuple[int, int]) -> tuple[Phase, ...]:
    """Intercala fases seguras. La red solamente aporta los tiempos de verde."""
    if greens == (0, 0):
        return (Phase('Rojo', 'Rojo', 0),)
    green_a, green_b = (bounded_green(seconds) for seconds in greens)
    clearance = Phase('Rojo', 'Rojo', ALL_RED)
    return (Phase('Verde', 'Rojo', green_a), Phase('Amarillo', 'Rojo', YELLOW), clearance,
            Phase('Rojo', 'Verde', green_b), Phase('Rojo', 'Amarillo', YELLOW), clearance)


def phase_at(cycle: tuple[Phase, ...], elapsed: int) -> tuple[Phase, int]:
    """Devuelve la fase activa y cuántos segundos faltan para el cambio."""
    duration = sum(phase.seconds for phase in cycle)
    if duration == 0:
        return cycle[0], 0

    position = max(0, elapsed) % duration
    for phase in cycle:
        if position < phase.seconds:
            return phase, phase.seconds - position
        position -= phase.seconds
    raise RuntimeError('El plan de fases no se pudo recorrer.')
