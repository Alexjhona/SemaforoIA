"""Escenarios de un cruce de dos vías y evaluación de planes en Eclipse SUMO."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
import subprocess
from xml.etree import ElementTree as ET

from traffic import ALL_RED, build_cycle


WARMUP = 45
HORIZON = 150
BACKLOG_WEIGHT = 20
GREEN_GRID = tuple(product(range(10, 36, 5), repeat=2))
INCOMING = ('a_in', 'b_in')
TLS = 'cruce'
COLORS = {'Verde': 'G', 'Amarillo': 'y', 'Rojo': 'r'}


def write_xml(path, root):
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def make_network(folder: Path) -> Path:
    """Construye dos movimientos rectos y un único cruce semaforizado."""
    folder.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element('nodes')
    for name, x, y in [('oeste', -280, 0), ('este', 65, 0), ('sur', 0, -280),
                       ('norte', 0, 65), (TLS, 0, 0)]:
        attrs = {'type': 'traffic_light'} if name == TLS else {}
        ET.SubElement(nodes, 'node', id=name, x=str(x), y=str(y), **attrs)
    edges = ET.Element('edges')
    for name, origin, destination in [('a_in', 'oeste', TLS), ('a_out', TLS, 'este'),
                                       ('b_in', 'sur', TLS), ('b_out', TLS, 'norte')]:
        ET.SubElement(edges, 'edge', {'id': name, 'from': origin, 'to': destination,
                                     'numLanes': '1', 'speed': '13.89'})
    links = ET.Element('connections')
    for approach in ('a', 'b'):
        ET.SubElement(links, 'connection', {'from': f'{approach}_in', 'to': f'{approach}_out',
                                            'fromLane': '0', 'toLane': '0'})
    for name, document in [('nodos', nodes), ('vias', edges), ('giros', links)]:
        write_xml(folder / f'{name}.xml', document)
    net = folder / 'cruce.net.xml'
    subprocess.run(['netconvert', '--node-files', str(folder / 'nodos.xml'),
                    '--edge-files', str(folder / 'vias.xml'), '--connection-files', str(folder / 'giros.xml'),
                    '--output-file', str(net)], check=True, capture_output=True, text=True)
    return net


def make_routes(folder: Path, scenario_seed: int) -> tuple[Path, int]:
    """Genera llegadas y mezcla de vehículos. Cada plan comparte las mismas llegadas."""
    rng = random.Random(scenario_seed)
    routes = ET.Element('routes')
    ET.SubElement(routes, 'vType', id='liviano', vClass='passenger', sigma='0.3')
    ET.SubElement(routes, 'vType', id='pesado', vClass='truck', length='9', accel='1.1', sigma='0.3')
    departures = []
    for approach in ('a', 'b'):
        ET.SubElement(routes, 'route', id=approach, edges=f'{approach}_in {approach}_out')
        demand = rng.uniform(0.025, 0.52)
        heavy_share = rng.uniform(0.03, 0.40)
        # El tramo posterior añade variación al flujo observable durante el calentamiento.
        future = min(0.60, demand * rng.uniform(0.65, 1.35))
        arrivals = [(0, WARMUP, demand), (WARMUP, WARMUP + HORIZON - 15, future)]
        for start, stop, probability in arrivals:
            for second in range(start, stop):
                if rng.random() < probability:
                    vehicle = 'pesado' if rng.random() < heavy_share else 'liviano'
                    departures.append((second, approach, vehicle))
    for index, (second, approach, vehicle) in enumerate(sorted(departures)):
        ET.SubElement(routes, 'vehicle', id=f'v{index}', type=vehicle, route=approach,
                      depart=str(second), departSpeed='max')
    path = folder / 'escenario.rou.xml'
    write_xml(path, routes)
    return path, len(departures)


@dataclass(frozen=True)
class SimulationResult:
    counts: tuple[int, int, int, int]
    queue_seconds: int
    arrived: int
    unfinished: int
    collision_events: int

    @property
    def objective(self):
        return self.queue_seconds + BACKLOG_WEIGHT * self.unfinished


class IntersectionSimulation:
    def __init__(self, net: Path):
        import libsumo

        self.sumo = libsumo
        self.net = net

    def run(self, routes: Path, scheduled: int, scenario_seed: int, greens: tuple[int, int]):
        sim = self.sumo
        sim.start(['sumo', '-n', str(self.net), '-r', str(routes), '--seed', str(scenario_seed),
                   '--no-step-log', '--no-warnings', '--duration-log.disable', '--time-to-teleport', '-1'])
        try:
            controlled = sim.trafficlight.getControlledLinks(TLS)
            order = [INCOMING.index(links[0][0].rsplit('_', 1)[0]) for links in controlled]
            if sorted(order) != [0, 1]:
                raise RuntimeError('SUMO debe exponer dos señales controladas, una por vía.')
            sim.trafficlight.setRedYellowGreenState(TLS, 'rr')
            for _ in range(WARMUP):
                sim.simulationStep()
            ids_a, ids_b = [sim.edge.getLastStepVehicleIDs(edge) for edge in INCOMING]
            counts = (len(ids_a), len(ids_b), sum(sim.vehicle.getTypeID(v) == 'pesado' for v in ids_a),
                      sum(sim.vehicle.getTypeID(v) == 'pesado' for v in ids_b))
            states = []
            for phase in build_cycle(greens):
                signal = (COLORS[phase.a], COLORS[phase.b])
                states.extend([''.join(signal[index] for index in order)] * phase.seconds)
            states = states or ['rr']
            # La primera autorización de paso también tiene un intervalo todo rojo.
            previous = 'rr'
            queue = arrived = collisions = 0
            for second in range(HORIZON):
                signal = 'rr' if second < ALL_RED else states[(second - ALL_RED) % len(states)]
                if signal != previous:
                    sim.trafficlight.setRedYellowGreenState(TLS, signal)
                    previous = signal
                sim.simulationStep()
                queue += sum(sim.edge.getLastStepHaltingNumber(edge) for edge in INCOMING)
                arrived += sim.simulation.getArrivedNumber()
                collisions += sim.simulation.getCollidingVehiclesNumber()
            return SimulationResult(counts, queue, arrived, scheduled - arrived, collisions)
        finally:
            sim.close()
