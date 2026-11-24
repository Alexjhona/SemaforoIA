"""Pruebas de SemaforoIA.

Ejecutar desde la carpeta del proyecto:
    python -m unittest discover -s tests -v

Las pruebas marcadas como «modelos reales» cargan best.pt y controlador_tiempos.pt
y se omiten automáticamente si faltan los pesos o las dependencias de inferencia.
"""

import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analisis import FrameSample, LaneCount, decide, saturate, video_decisions, window_count  # noqa: E402
from detector import DetectionSettings, VehicleDetector, Zone, decode_image  # noqa: E402
from traffic import Phase, build_cycle, demonstration_greens, phase_at  # noqa: E402


def has_real_models() -> bool:
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return (ROOT / 'models' / 'best.pt').is_file() and (ROOT / 'models' / 'controlador_tiempos.pt').is_file()


class FakeController:
    """Controlador de prueba: solo para verificar el flujo, no para la app."""
    count_limits = (23, 25)

    def __init__(self):
        self.calls = []

    def predict(self, a, b, ha, hb):
        self.calls.append((a, b, ha, hb))
        return (0, 0) if a + b == 0 else (10 + a, 10 + b)


class TrafficTests(unittest.TestCase):
    def test_empty_intersection(self):
        self.assertEqual(phase_at(build_cycle((0, 0)), 900), (Phase('Rojo', 'Rojo', 0), 0))

    def test_green_bounds_and_clearance(self):
        cycle = build_cycle((2, 70))
        self.assertEqual(cycle[0], Phase('Verde', 'Rojo', 10))
        self.assertEqual(cycle[2], Phase('Rojo', 'Rojo', 2))
        self.assertEqual(cycle[3], Phase('Rojo', 'Verde', 35))
        self.assertEqual(sum(phase.seconds for phase in cycle), 55)

    def test_no_simultaneous_green_through_cycle(self):
        for count_a, count_b in [(1, 1), (60, 1), (1, 60), (5, 0), (0, 5)]:
            cycle = build_cycle(demonstration_greens(count_a, count_b))
            for seconds in range(110):
                phase, remaining = phase_at(cycle, seconds)
                self.assertIn('Rojo', (phase.a, phase.b))
                self.assertGreaterEqual(remaining, 1)


class DetectorTests(unittest.TestCase):
    def test_decode_and_reject_invalid_file(self):
        buffer = BytesIO()
        Image.new('L', (12, 8)).save(buffer, format='PNG')
        self.assertEqual(decode_image(buffer.getvalue()).mode, 'RGB')
        with self.assertRaises(ValueError):
            decode_image(b'archivo que no es una imagen')

    def _fake(self, boxes):
        fake_yolo = Mock()
        fake_yolo.names = {0: 'car', 1: 'per', 2: 'truck'}
        result = Mock()
        result.boxes.xyxy.tolist.return_value = [b[:4] for b in boxes]
        result.boxes.cls.tolist.return_value = [b[4] for b in boxes]
        result.boxes.conf.tolist.return_value = [0.9] * len(boxes)
        result.speed = {'inference': 12.0}
        fake_yolo.predict.return_value = [result]
        module = Mock()
        module.YOLO.return_value = fake_yolo
        return module, fake_yolo

    def test_vehicle_filter_counts_and_heavy(self):
        module, fake_yolo = self._fake([(0, 0, 10, 10, 0), (20, 20, 30, 30, 0), (40, 40, 60, 60, 2)])
        with patch.dict('sys.modules', {'ultralytics': module}):
            detector = VehicleDetector(Path('models/best.pt'))
            out = detector.predict([Image.new('RGB', (64, 64))], settings=DetectionSettings())[0]
        self.assertEqual(fake_yolo.predict.call_args.kwargs['classes'], [0, 2])
        self.assertEqual(out.counts, {'Auto': 2, 'Camión': 1})
        self.assertEqual((out.total, out.heavy), (3, 1))

    def test_nested_boxes_and_zone(self):
        # Una "rueda" (caja pequeña dentro de un camión) y un auto fuera de la zona inferior.
        module, _ = self._fake([(0, 40, 60, 64, 2), (5, 50, 12, 60, 0), (10, 2, 20, 12, 0)])
        with patch.dict('sys.modules', {'ultralytics': module}):
            detector = VehicleDetector(Path('models/best.pt'))
            settings = DetectionSettings(zone=Zone(0, 0.5, 1, 1))
            out = detector.predict([Image.new('RGB', (64, 64))], settings=settings)[0]
        self.assertEqual(out.counts, {'Camión': 1})


class AnalysisTests(unittest.TestCase):
    def test_saturation_keeps_heavy_share(self):
        self.assertEqual(saturate(10, 2, 23), (10, 2, False))
        self.assertEqual(saturate(46, 10, 23), (23, 5, True))

    def test_decide_feeds_mlp_and_builds_safe_cycle(self):
        controller = FakeController()
        decision = decide(controller, LaneCount(30, 4), LaneCount(5, 1))
        self.assertEqual(controller.calls, [(23, 5, 3, 1)])
        self.assertEqual(decision.greens, (33, 15))
        self.assertEqual(decision.saturated, (True, False))
        self.assertEqual([p['s'] for p in decision.phases], [33, 3, 2, 15, 3, 2])
        raw = decide(FakeController(), LaneCount(30, 4), LaneCount(5, 1), use_saturation=False)
        self.assertEqual(raw.mlp_input, (30, 5, 4, 1))

    def test_windows_use_median_and_loop_short_video(self):
        a = [FrameSample(t, LaneCount(n, 0, {'Auto': n}), b'') for t, n in [(0, 4), (1, 50), (2, 6), (5, 9)]]
        b = [FrameSample(t, LaneCount(2, 1), b'') for t in (0, 1)]
        self.assertEqual(window_count(a, 0, 5, 6).total, 6)
        decisions = video_decisions(FakeController(), a, b, 6.0, 2.0, window=3)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[1].detected[1].total, 2)  # B (2 s) se repite en bucle


class VideoTests(unittest.TestCase):
    def test_sampling_frequency(self):
        import cv2
        from video import count_samples, sample_frames, video_info

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prueba.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            for i in range(30):
                writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
            writer.release()
            self.assertEqual(video_info(path).frames, 30)
            frames = list(sample_frames(path, samples_per_second=2, max_seconds=2))
            self.assertEqual([round(t, 1) for t, _ in frames], [0.0, 0.5, 1.0, 1.5])
            self.assertEqual(count_samples(path, 2, 2), 4)
            self.assertEqual(frames[0][1].shape, (48, 64, 3))


class SimulatorTests(unittest.TestCase):
    def test_html_contains_decisions(self):
        import simulador

        decision = decide(FakeController(), LaneCount(3, 0), LaneCount(1, 0))
        jpeg = simulador.to_jpeg(Image.new('RGB', (800, 400)))
        html = simulador.build_html(simulador.lane_payload('Avenida A', 'a.jpg', [(0, 3, jpeg)], 0),
                                    simulador.lane_payload('Avenida B', 'b.jpg', [], 0), [decision], None)
        self.assertNotIn('__DATA__', html)
        self.assertIn('"greens": [13, 11]', html)
        self.assertIn('btnStart', html)


class LiveTests(unittest.TestCase):
    def test_signal_follows_safe_phases_and_asks_mlp_each_cycle(self):
        from en_vivo import LiveSignal

        controller = FakeController()
        signal = LiveSignal(controller)
        lane_a, lane_b = LaneCount(5, 0), LaneCount(2, 0)  # FakeController → verdes 15 y 12
        self.assertEqual((signal.update(0.5, lane_a, lane_b).a, signal.decision), ('Rojo', None))
        first = signal.update(2.0, lane_a, lane_b)
        self.assertEqual((first.a, first.b, first.new_decision), ('Verde', 'Rojo', True))
        self.assertAlmostEqual(first.remaining_b, 15 + 3 + 2)
        self.assertEqual(signal.update(2 + 16, lane_a, lane_b).a, 'Amarillo')
        self.assertEqual(signal.update(2 + 21, lane_a, lane_b).b, 'Verde')
        for t in range(0, 200):  # nunca dos verdes a la vez
            state = signal.update(2 + t * 0.7, LaneCount(t % 20, 0), LaneCount(3, 1))
            self.assertIn('Rojo', (state.a, state.b))
        self.assertGreater(len(controller.calls), 3)

    def test_source_loops_video_and_reads_photo(self):
        import cv2
        from en_vivo import LiveSource

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cam.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            for i in range(20):
                writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
            writer.release()
            source = LiveSource(path)
            self.assertAlmostEqual(source.duration, 2.0)
            self.assertAlmostEqual(source.frame_at(1.5)[0], 1.5)
            self.assertAlmostEqual(source.frame_at(2.3)[0], 0.3)  # volvió a empezar
            source.close()
            photo = Path(folder) / 'foto.png'
            Image.new('RGB', (40, 30)).save(photo)
            self.assertEqual(LiveSource(photo).frame_at(99)[1].shape, (30, 40, 3))


@unittest.skipUnless(has_real_models(), 'modelos reales o dependencias no disponibles')
class RealModelTests(unittest.TestCase):
    """Usa best.pt y controlador_tiempos.pt sin modificarlos."""

    @classmethod
    def setUpClass(cls):
        import modelos

        cls.detector, cls.yolo_status = modelos.load_yolo()
        cls.controller, cls.mlp_status = modelos.load_mlp()

    def test_models_load(self):
        self.assertTrue(self.yolo_status.loaded, self.yolo_status.error)
        self.assertTrue(self.mlp_status.loaded, self.mlp_status.error)
        self.assertEqual(set(self.detector.labels.values()), {'Auto', 'Bus', 'Camión', 'Motocicleta', 'Bicicleta'})

    def test_example_photo_to_green_times(self):
        photo = ROOT / 'ejemplos' / 'fotos' / 'avenida_A_foto1.jpg'
        if not photo.is_file():
            self.skipTest('sin foto de ejemplo')
        det = self.detector.predict([Image.open(photo)], settings=DetectionSettings())[0]
        self.assertGreater(det.total, 0)
        decision = decide(self.controller, LaneCount(det.total, det.heavy), LaneCount(0, 0))
        self.assertTrue(10 <= decision.greens[0] <= 35)
        self.assertEqual(decision.greens, self.controller.predict(*decision.mlp_input))

    def test_mlp_is_deterministic_and_bounded(self):
        for counts in [(5, 5, 0, 0), (20, 3, 2, 0), (3, 20, 0, 3)]:
            greens = self.controller.predict(*counts)
            self.assertEqual(greens, self.controller.predict(*counts))
            self.assertTrue(all(10 <= g <= 35 for g in greens))


class AppSmokeTest(unittest.TestCase):
    @unittest.skipUnless(has_real_models(), 'modelos reales o dependencias no disponibles')
    def test_app_starts_without_exceptions(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        # La pestaña de prueba directa muestra la predicción de la MLP con 15 y 6 vehículos.
        values = [m.value for m in app.metric]
        self.assertIn('15 veh.', values)


if __name__ == '__main__':
    unittest.main()
