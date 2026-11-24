"""Diagnóstico rápido del equipo: versiones, carga de modelos y tiempos en CPU.

Uso:  .\\.venv\\Scripts\\python.exe diagnostico.py
No entrena ni modifica los modelos; solo los lee y mide tiempos de inferencia.
"""

import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    print('=== SemaforoIA · diagnóstico ===')
    print('Python', sys.version.split()[0], '·', platform.platform())
    import cv2
    import streamlit
    import torch
    import ultralytics

    print('torch', torch.__version__, '· CUDA disponible:', torch.cuda.is_available())
    print('ultralytics', ultralytics.__version__, '· streamlit', streamlit.__version__, '· opencv', cv2.__version__)

    import modelos
    from PIL import Image

    from analisis import FrameSample, LaneCount, decide, video_decisions
    from detector import DetectionSettings
    from video import sample_frames, video_info

    detector, yolo = modelos.load_yolo()
    controller, mlp = modelos.load_mlp()
    for status in (yolo, mlp):
        print(f'{status.name}: {"OK" if status.loaded else "ERROR " + status.error} '
              f'({status.load_seconds:.2f} s)', status.details)
    if not (yolo.loaded and mlp.loaded):
        sys.exit(1)

    settings = DetectionSettings()
    photos = sorted((ROOT / 'ejemplos' / 'fotos').glob('*.jpg'))
    for photo in photos:
        start = perf_counter()
        det = detector.predict([Image.open(photo)], settings=settings)[0]
        print(f'Foto {photo.name}: {det.counts} total={det.total} pesados={det.heavy} '
              f'· {1000 * (perf_counter() - start):.0f} ms')
    if len(photos) >= 3:
        a = detector.predict([Image.open(photos[0])], settings=settings)[0]
        b = detector.predict([Image.open(photos[2])], settings=settings)[0]
        decision = decide(controller, LaneCount(a.total, a.heavy), LaneCount(b.total, b.heavy))
        print(f'MLP con {photos[0].name} y {photos[2].name}: entrada {decision.mlp_input} → verdes {decision.greens} s')

    lanes = []
    for name in ('avenida_A_trafico_denso.mp4', 'avenida_B_buses_y_autos.mp4'):
        path = ROOT / 'ejemplos' / 'videos' / name
        if not path.is_file():
            continue
        info = video_info(path)
        start, samples = perf_counter(), []
        for second, frame in sample_frames(path, 2, 20):
            det = detector.predict([frame], settings=settings)[0]
            samples.append(FrameSample(second, LaneCount(det.total, det.heavy, det.counts), b''))
        elapsed = perf_counter() - start
        print(f'Video {name}: {info.seconds:.1f} s, {len(samples)} fotogramas en {elapsed:.1f} s '
              f'({len(samples) / elapsed:.1f} fps) · conteos {[s.count.total for s in samples]}')
        lanes.append((samples, info.seconds))
    if len(lanes) == 2:
        for d in video_decisions(controller, lanes[0][0], lanes[1][0], lanes[0][1], lanes[1][1], 5):
            print(f'  ventana {d.label}: detectados A={d.detected[0].total} B={d.detected[1].total} '
                  f'→ MLP verdes {d.greens} s, ciclo {d.cycle_seconds} s')
    print('=== Diagnóstico terminado sin errores ===')


if __name__ == '__main__':
    main()
