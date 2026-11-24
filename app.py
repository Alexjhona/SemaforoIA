"""SemaforoIA · Aplicación web (Streamlit).

Ejecutar desde la carpeta del proyecto:
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

Etapas de la aplicación
  1. Carga única de los modelos entrenados (best.pt y controlador_tiempos.pt).
  2. Entrada: dos fotografías o dos videos, una por avenida.
  3. YOLOv8n detecta y cuenta vehículos por categoría (detector.py).
  4. La MLP convierte los conteos en segundos de verde (analisis.py + controller.py).
  5. El plan de fases seguro se reproduce en dos semáforos animados (simulador.py).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from time import monotonic, perf_counter, sleep

import streamlit as st

import modelos
import simulador
from en_vivo import CountHistory, LiveSignal, LiveSource, signal_html
from analisis import FrameSample, LaneCount, decide, video_decisions
from detector import CATEGORY_ORDER, DetectionSettings, Zone, decode_image
from video import count_samples, sample_frames, video_info


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / 'ejemplos'
PHOTO_TYPES = ['jpg', 'jpeg', 'png', 'webp']
VIDEO_TYPES = ['mp4', 'avi', 'mov', 'mkv']

st.set_page_config(page_title='SemaforoIA · Control semafórico inteligente', page_icon='🚦', layout='wide')
st.markdown(f'<style>{(ROOT / "style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA ÚNICA DE LOS MODELOS
#    st.cache_resource guarda el modelo en memoria: se carga una sola vez aunque
#    la página se recargue o haya varios usuarios. La fecha del archivo forma parte
#    de la clave, así un peso reemplazado se vuelve a cargar automáticamente.
# ─────────────────────────────────────────────────────────────────────────────
def _mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.is_file() else 0


@st.cache_resource(show_spinner='Cargando YOLOv8n (best.pt)…')
def get_detector(modified: int):
    return modelos.load_yolo()


@st.cache_resource(show_spinner='Cargando la MLP (controlador_tiempos.pt)…')
def get_controller(modified: int):
    return modelos.load_mlp()


detector, yolo_status = get_detector(_mtime(modelos.YOLO_WEIGHTS))
controller, mlp_status = get_controller(_mtime(modelos.MLP_WEIGHTS))
metrics = modelos.training_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Componentes visuales reutilizables
# ─────────────────────────────────────────────────────────────────────────────
def status_card(status: modelos.ModelStatus, extra: str = '') -> str:
    if status.loaded:
        rows = ''.join(f'<li><span>{k}</span><b>{v}</b></li>' for k, v in status.details.items())
        return (f'<div class="model-card ok"><div class="model-title"><i></i>{status.name}</div>'
                f'<div class="model-sub">{status.path.name} · {status.size_mb:.2f} MB · '
                f'cargado en {status.load_seconds:.1f} s</div><ul>{rows}{extra}</ul></div>')
    return (f'<div class="model-card bad"><div class="model-title"><i></i>{status.name}</div>'
            f'<div class="model-sub">{status.error}</div></div>')


def category_chips(counts: dict[str, int]) -> str:
    chips = ''.join(f'<span class="chip cat-{i}"><b>{counts.get(name, 0)}</b> {name}</span>'
                    for i, name in enumerate(CATEGORY_ORDER))
    return f'<div class="chips">{chips}</div>'


def lane_header(title: str, total, heavy, suffix: str = '') -> str:
    return (f'<div class="lane-title"><h3>{title}</h3><div><span class="pill">{total} vehículos{suffix}</span>'
            f'<span class="pill heavy">{heavy} pesados{suffix}</span></div></div>')


def decision_summary(decision, title: str = 'Decisión de la red neuronal') -> None:
    """Muestra entradas y salidas de la MLP con advertencias honestas."""
    st.markdown(f'#### {title}')
    ia, ib, ha, hb = decision.mlp_input
    cols = st.columns(4)
    cols[0].metric(f'Entrada A · {ha} pesados', f'{ia} veh.')
    cols[1].metric(f'Entrada B · {hb} pesados', f'{ib} veh.')
    cols[2].metric('Verde Avenida A', f'{decision.greens[0]} s')
    cols[3].metric('Verde Avenida B', f'{decision.greens[1]} s')
    if decision.greens == (0, 0):
        st.info('No se detectaron vehículos en ninguna avenida: ambas permanecen en rojo.')
    for lane, sat, real in (('A', decision.saturated[0], decision.detected[0].total),
                            ('B', decision.saturated[1], decision.detected[1].total)):
        if sat:
            st.warning(f'Avenida {lane}: hay {real} vehículos y la MLP se entrenó hasta '
                       f'{controller.count_limits[0 if lane == "A" else 1]}. Se usa el máximo entrenado '
                       '(avenida saturada). Puedes reducir la zona de detección al tramo cercano al cruce.')
    limits = controller.count_limits
    if not any(decision.saturated) and (decision.mlp_input[0] > limits[0] or decision.mlp_input[1] > limits[1]):
        st.warning('El conteo supera el rango de entrenamiento de la MLP; la predicción es una extrapolación.')


def example_files(kind: str, suffixes: tuple[str, ...]) -> list[Path]:
    folder = EXAMPLES / kind
    return sorted(p for p in folder.glob('*') if p.suffix.lower() in suffixes) if folder.is_dir() else []


def nice(path: Path) -> str:
    text = path.stem.replace('_', ' ')
    return text[:1].upper() + text[1:]


# ─────────────────────────────────────────────────────────────────────────────
# 2. BARRA LATERAL: estado de los modelos y parámetros
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### Modelos')
    yolo_extra = mlp_extra = ''
    if metrics.get('yolo'):
        yolo_extra = f'<li><span>mAP50 (prueba)</span><b>{metrics["yolo"]["mAP50"]:.1%}</b></li>'
    if metrics.get('mlp'):
        mlp_extra = (f'<li><span>Error medio (MAE)</span><b>A {metrics["mlp"]["mae_a_s"]:.1f} s · '
                     f'B {metrics["mlp"]["mae_b_s"]:.1f} s</b></li>')
    st.markdown(status_card(yolo_status, yolo_extra) + status_card(mlp_status, mlp_extra), unsafe_allow_html=True)

    advanced = st.expander('Ajustes avanzados', expanded=False)
    advanced.markdown('**Detección (YOLOv8n)**')
    confidence = advanced.slider('Umbral de confianza', 0.05, 0.90, 0.25, 0.05,
                           help='Solo se cuentan las detecciones con una confianza mayor o igual a este valor.')
    image_size = advanced.select_slider('Resolución de inferencia (px)', [320, 416, 512, 640], value=640,
                                  help='640 es la resolución de entrenamiento. Valores menores son más rápidos en CPU.')
    agnostic = advanced.toggle('Una sola etiqueta por vehículo', value=True,
                         help='NMS agnóstico: evita contar el mismo vehículo como Auto y como Camión.')
    nested = advanced.toggle('Ignorar cajas dentro de otra', value=True,
                       help='Descarta detecciones pequeñas encerradas en un vehículo mayor (ruedas, ventanas).')
    use_zone = advanced.toggle('Limitar a una zona de detección', value=False,
                         help='Cuenta solo los vehículos cuyo centro está en el tramo seleccionado (acceso al cruce).')
    zone = Zone()
    if use_zone:
        top, bottom = advanced.slider('Franja vertical (%)', 0, 100, (35, 100), 5)
        left, right = advanced.slider('Franja horizontal (%)', 0, 100, (0, 100), 5)
        if bottom - top >= 5 and right - left >= 5:
            zone = Zone(left / 100, top / 100, right / 100, bottom / 100)
    settings = DetectionSettings(confidence=confidence, image_size=image_size, zone=zone,
                                 agnostic_nms=agnostic, drop_nested=nested)

    advanced.markdown('**Video**')
    fps_sample = advanced.select_slider('Fotogramas analizados por segundo', [0.5, 1.0, 2.0, 3.0, 5.0], value=2.0,
                                  help='Frecuencia de procesamiento. Menos fotogramas = análisis más rápido en CPU.')
    max_seconds = advanced.slider('Duración máxima analizada (s)', 5, 60, 20, 5)
    window = advanced.slider('Ventana de decisión (s)', 2, 15, 5, 1,
                       help='Cada cuántos segundos de video la MLP recalcula los tiempos de verde.')

    advanced.markdown('**Controlador (MLP)**')
    use_saturation = advanced.toggle('Saturar conteos fuera del rango entrenado', value=True,
                               help='Si YOLO cuenta más vehículos de los vistos en el entrenamiento, se usa el máximo entrenado.')


# ─────────────────────────────────────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>SemaforoIA</h1>
  <p>Semáforo inteligente: YOLOv8n cuenta los vehículos y una red neuronal decide los segundos de verde.</p>
</div>
""", unsafe_allow_html=True)

models_ready = detector is not None and controller is not None
if not models_ready:
    st.error('Los dos modelos deben estar cargados para analizar. Revisa el panel «Estado de los modelos».')

# La pestaña «En vivo» se muestra primero, pero su código va al final del archivo:
# así, mientras la cámara corre, las demás pestañas ya están dibujadas.
tab_live, tab_photo, tab_video, tab_manual, tab_help = st.tabs(
    ['En vivo', 'Fotos', 'Videos', 'Probar la red', 'Cómo funciona'])


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANÁLISIS DE FOTOGRAFÍAS
# ─────────────────────────────────────────────────────────────────────────────
def pick_inputs(key: str, kind: str, types: list[str], suffixes: tuple[str, ...]):
    """Devuelve [(nombre, bytes o ruta) para A, para B] según el origen elegido."""
    examples = example_files(kind, suffixes)
    options = ['Subir mis archivos'] + (['Usar ejemplos incluidos'] if examples else [])
    origin = st.radio('Origen', options, horizontal=True, key=f'{key}_origin', label_visibility='collapsed')
    col_a, col_b = st.columns(2)
    picked = []
    default_b = next((i for i, p in enumerate(examples) if '_B_' in p.name), min(1, len(examples) - 1))
    for column, lane, default in ((col_a, 'A', 0), (col_b, 'B', default_b)):
        with column:
            if origin == 'Subir mis archivos':
                file = st.file_uploader(f'Avenida {lane}', type=types, key=f'{key}_{lane}')
                picked.append((file.name, file.getvalue()) if file else None)
            else:
                path = st.selectbox(f'Avenida {lane}', examples, index=max(0, default), format_func=nice,
                                    key=f'{key}_ex_{lane}')
                picked.append((path.name, path))
    return picked


with tab_photo:
    inputs = pick_inputs('foto', 'fotos', PHOTO_TYPES, ('.jpg', '.jpeg', '.png', '.webp'))
    if st.button('Analizar fotos', type='primary', disabled=not models_ready,
                 key='btn_foto'):
        if None in inputs:
            st.warning('Falta una fotografía: selecciona una imagen para cada avenida.')
        else:
            try:
                images = [decode_image(src.read_bytes() if isinstance(src, Path) else src) for _, src in inputs]
                start = perf_counter()
                with st.spinner('YOLOv8n está buscando vehículos en las dos avenidas…'):
                    detections = detector.predict(images, settings=settings)  # ← inferencia real de best.pt
                st.session_state.foto = {'names': [name for name, _ in inputs], 'detections': detections,
                                         'seconds': perf_counter() - start, 'settings': settings}
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f'No se pudo analizar el par de imágenes. {error}')

    result = st.session_state.get('foto')
    if result:
        if result['settings'] != settings:
            st.caption('Cambiaste los ajustes de detección: pulsa «Analizar» para aplicarlos.')
        det_a, det_b = result['detections']
        columns = st.columns(2)
        for column, title, det, name in zip(columns, ('Avenida A', 'Avenida B'), (det_a, det_b), result['names']):
            with column:
                st.markdown(lane_header(title, det.total, det.heavy), unsafe_allow_html=True)
                st.image(det.image, caption=f'{name} · inferencia {det.inference_ms:.0f} ms', width='stretch')
                st.markdown(category_chips(det.counts), unsafe_allow_html=True)
        st.caption(f'Analizado en {result["seconds"]:.2f} s.')

        # La MLP se ejecuta en cada recarga (es instantánea): así responde a los ajustes del controlador.
        lane_a = LaneCount(det_a.total, det_a.heavy, det_a.counts)
        lane_b = LaneCount(det_b.total, det_b.heavy, det_b.counts)
        decision = decide(controller, lane_a, lane_b, use_saturation)
        decision_summary(decision)

        st.markdown('#### Semáforos')
        simulador.show(
            simulador.lane_payload('Avenida A', result['names'][0], [(0.0, det_a.total, simulador.to_jpeg(det_a.image))], 0),
            simulador.lane_payload('Avenida B', result['names'][1], [(0.0, det_b.total, simulador.to_jpeg(det_b.image))], 0),
            [decision])


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANÁLISIS DE VIDEOS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_video(source, lane: str, progress, offset: int, total: int) -> dict:
    """Recorre el video con la frecuencia elegida y guarda un FrameSample por fotograma analizado."""
    temp = None
    if isinstance(source, Path):
        path = source
    else:  # archivo subido: se escribe en un temporal que se borra al terminar
        temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp.write(source)
        temp.close()
        path = Path(temp.name)
    try:
        info = video_info(path)
        samples, start = [], perf_counter()
        for i, (second, frame) in enumerate(sample_frames(path, fps_sample, max_seconds)):
            det = detector.predict([frame], settings=settings)[0]  # ← YOLOv8n sobre el fotograma
            samples.append(FrameSample(second, LaneCount(det.total, det.heavy, det.counts),
                                       simulador.to_jpeg(det.image, max_width=640, quality=68)))
            progress.progress(min(1.0, (offset + i + 1) / total),
                              text=f'Avenida {lane}: fotograma {i + 1} · t = {second:.1f} s · {det.total} vehículos')
        duration = min(info.seconds, max_seconds) if info.seconds else (samples[-1].time if samples else 0)
        return {'samples': samples, 'duration': max(duration, 0.5), 'seconds': perf_counter() - start, 'info': info}
    finally:
        if temp is not None:
            Path(temp.name).unlink(missing_ok=True)


with tab_video:
    inputs = pick_inputs('video', 'videos', VIDEO_TYPES, ('.mp4', '.avi', '.mov', '.mkv'))
    if st.button('Analizar videos', type='primary', disabled=not models_ready, key='btn_video'):
        if None in inputs:
            st.warning('Falta un video: selecciona un archivo para cada avenida.')
        else:
            try:
                st.session_state.pop('video', None)
                # Se escriben temporales solo para contar los fotogramas de la barra de progreso.
                totals = []
                for _, src in inputs:
                    if isinstance(src, Path):
                        totals.append(count_samples(src, fps_sample, max_seconds))
                    else:
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                            tmp.write(src)
                        totals.append(count_samples(Path(tmp.name), fps_sample, max_seconds))
                        Path(tmp.name).unlink(missing_ok=True)
                bar = st.progress(0.0, text='Preparando el análisis…')
                lane_a = analyze_video(inputs[0][1], 'A', bar, 0, sum(totals))
                lane_b = analyze_video(inputs[1][1], 'B', bar, totals[0], sum(totals))
                bar.empty()
                if not lane_a['samples'] or not lane_b['samples']:
                    st.error('No se pudieron leer fotogramas de uno de los videos.')
                else:
                    st.session_state.video = {'names': [n for n, _ in inputs], 'A': lane_a, 'B': lane_b,
                                              'settings': (settings, fps_sample, max_seconds)}
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f'No se pudo analizar el video. {error}')

    result = st.session_state.get('video')
    if result:
        if result['settings'] != (settings, fps_sample, max_seconds):
            st.caption('Cambiaste los ajustes de detección o muestreo: pulsa «Analizar» para aplicarlos.')
        lanes = (result['A'], result['B'])
        cols = st.columns(2)
        for column, title, lane, name in zip(cols, ('Avenida A', 'Avenida B'), lanes, result['names']):
            totals = [s.count.total for s in lane['samples']]
            with column:
                peak = max(lane['samples'], key=lambda s: s.count.total)
                st.markdown(lane_header(title, f'{sum(totals) / len(totals):.1f}',
                                        f'{sum(s.count.heavy for s in lane["samples"]) / len(totals):.1f}',
                                        ' (promedio)'), unsafe_allow_html=True)
                st.image(peak.jpeg, caption=f'{name} · fotograma con más vehículos (t = {peak.time:.1f} s)',
                         width='stretch')
                st.markdown(category_chips(peak.count.by_class), unsafe_allow_html=True)
                n = len(lane['samples'])
                st.caption(f'{n} fotogramas analizados de {lane["duration"]:.1f} s de video · '
                           f'{lane["seconds"]:.1f} s de proceso ({n / max(lane["seconds"], 1e-6):.1f} fotogramas/s) · '
                           f'promedio {sum(totals) / n:.1f} y máximo {max(totals)} vehículos')

        # Evolución del conteo en el tiempo (datos reales de YOLO por fotograma).
        import pandas as pd

        series = {f'Avenida {k}': pd.Series({round(s.time, 1): s.count.total for s in lane['samples']})
                  for k, lane in zip('AB', lanes)}
        st.markdown('#### Vehículos detectados en el tiempo')
        st.line_chart(pd.DataFrame(series), x_label='segundo del video', y_label='vehículos')

        decisions = video_decisions(controller, lanes[0]['samples'], lanes[1]['samples'],
                                    lanes[0]['duration'], lanes[1]['duration'], window, use_saturation)
        st.markdown('#### Decisiones de la red por tramo de video')
        st.dataframe([{'Ventana': d.label,
                       'Vehículos A (mediana)': d.detected[0].total, 'Pesados A': d.detected[0].heavy,
                       'Vehículos B (mediana)': d.detected[1].total, 'Pesados B': d.detected[1].heavy,
                       'Entrada MLP (A, B)': f'{d.mlp_input[0]}, {d.mlp_input[1]}',
                       'Verde A (s)': d.greens[0], 'Verde B (s)': d.greens[1], 'Ciclo (s)': d.cycle_seconds}
                      for d in decisions], hide_index=True, width='stretch')
        if any(any(d.saturated) for d in decisions):
            st.warning('En algunas ventanas el conteo superó el rango de entrenamiento de la MLP y se saturó.')

        st.markdown('#### Semáforos')
        simulador.show(
            simulador.lane_payload('Avenida A', result['names'][0],
                                   [(s.time, s.count.total, s.jpeg) for s in lanes[0]['samples']], lanes[0]['duration']),
            simulador.lane_payload('Avenida B', result['names'][1],
                                   [(s.time, s.count.total, s.jpeg) for s in lanes[1]['samples']], lanes[1]['duration']),
            decisions, float(window))


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRUEBA DIRECTA DE LA MLP (conteos escritos a mano, sin YOLO)
# ─────────────────────────────────────────────────────────────────────────────
with tab_manual:
    st.caption('Conteos escritos a mano. Los segundos de verde los calcula la red entrenada.')
    if controller is None:
        st.error('La MLP no está cargada.')
    else:
        limit_a, limit_b = controller.count_limits
        c1, c2, c3, c4 = st.columns(4)
        va = c1.number_input('Vehículos · Avenida A', 0, 80, 15)
        vb = c2.number_input('Vehículos · Avenida B', 0, 80, 6)
        ha = c3.number_input('Pesados · Avenida A', 0, int(va), min(2, int(va)))
        hb = c4.number_input('Pesados · Avenida B', 0, int(vb), min(1, int(vb)))
        st.caption(f'Rango visto en el entrenamiento: Avenida A hasta {limit_a} y Avenida B hasta {limit_b} vehículos.')
        manual = decide(controller, LaneCount(int(va), int(ha)), LaneCount(int(vb), int(hb)), use_saturation,
                        label='Conteo manual')
        decision_summary(manual, 'Respuesta de la MLP al conteo manual')
        simulador.show(simulador.lane_payload('Avenida A', 'conteo manual', [], 0),
                       simulador.lane_payload('Avenida B', 'conteo manual', [], 0), [manual])


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPLICACIÓN PARA LA EXPOSICIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab_help:
    st.markdown("""
### ¿Cómo decide SemaforoIA?

| Etapa | Qué hace | Archivo |
|---|---|---|
| 1. Entrada | Una foto o un video por avenida. En video se analizan *N* fotogramas por segundo. | `video.py` |
| 2. Detección | **YOLOv8n** (`best.pt`) encuentra autos, motocicletas, camiones, buses y bicicletas y dibuja sus cuadros. | `detector.py` |
| 3. Conteo | Se cuentan los vehículos por avenida y categoría. Camión + Bus = vehículos **pesados**. | `detector.py` |
| 4. Decisión | La **MLP** (`controlador_tiempos.pt`) recibe 6 variables y predice los segundos de verde de A y B (10–35 s). | `controller.py`, `analisis.py` |
| 5. Seguridad | Verde → amarillo 3 s → rojo total 2 s → verde de la otra avenida. Nunca hay dos verdes. | `traffic.py` |
| 6. Simulación | Dos semáforos animados con temporizadores, controlados con Iniciar / Detener / Reiniciar. | `simulador.py` |

**Entradas de la MLP:** vehículos A, vehículos B, pesados A, pesados B, demanda total y proporción de A.
Arquitectura 6 → 64 → 32 → 2 con ReLU y una salida sigmoide escalada a 10–35 s.
Se entrenó con 192 escenarios simulados en **Eclipse SUMO**, donde la etiqueta fue el par de verdes con menor espera.
""")
    if metrics:
        y, m = metrics.get('yolo', {}), metrics.get('mlp', {})
        c = st.columns(4)
        c[0].metric('YOLO · mAP50 (prueba)', f'{y.get("mAP50", 0):.1%}')
        c[1].metric('YOLO · Precisión', f'{y.get("precision", 0):.1%}')
        c[2].metric('MLP · MAE Avenida A', f'{m.get("mae_a_s", 0):.2f} s')
        c[3].metric('MLP · MAE Avenida B', f'{m.get("mae_b_s", 0):.2f} s')
        st.caption('Métricas reales obtenidas en Colab (prueba de 671 imágenes para YOLO y 29 escenarios para la MLP).')
    per_class = ROOT / 'resultados' / 'v1_semilla42' / 'evaluacion' / 'prueba_final' / 'metricas_por_clase.csv'
    if per_class.is_file():
        import pandas as pd

        table = pd.read_csv(per_class).rename(columns={'clase': 'Clase', 'precision': 'Precisión',
                                                       'recall': 'Sensibilidad'})
        st.markdown('**Desempeño de YOLOv8n por clase (prueba oficial)**')
        st.dataframe(table, hide_index=True, width='stretch')
    st.caption('Nota: el detector se entrenó con cámaras elevadas de una intersección '
            '(dataset Intersection-traffic-detection). Funciona mejor con vistas altas donde los vehículos se ven '
            'pequeños; en primeros planos puede fragmentar o confundir vehículos.')


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODO EN VIVO: la cámara corre, YOLO detecta y los semáforos cambian en tiempo real
# ─────────────────────────────────────────────────────────────────────────────
def live_file(lane: str, upload) -> Path:
    """Guarda el archivo subido una sola vez en una carpeta temporal y devuelve su ruta."""
    store = st.session_state.setdefault('live_files', {})
    key = (lane, upload.name, upload.size)
    if key not in store or not store[key].is_file():
        folder = Path(tempfile.mkdtemp(prefix='semaforoia_'))
        path = folder / f'avenida_{lane}{Path(upload.name).suffix.lower()}'
        path.write_bytes(upload.getvalue())
        store[key] = path
    return store[key]


def run_live(paths: list[Path], target_fps: float, slots: dict) -> None:
    """Bucle en tiempo real. Termina al pulsar Detener (Streamlit reinicia el script)."""
    sources = [LiveSource(path) for path in paths]
    histories = [CountHistory(3.0), CountHistory(3.0)]
    signal = LiveSignal(controller, use_saturation)
    start = monotonic()
    try:
        while monotonic() - start < 15 * 60:          # límite de seguridad: 15 minutos
            tick = monotonic()
            t = tick - start
            frames = [source.frame_at(t) for source in sources]
            # 1) YOLOv8n analiza el fotograma actual de cada avenida (en un solo lote).
            detections = detector.predict([frame for _, frame in frames], settings=settings)
            for history, det in zip(histories, detections):
                history.add(t, det.total, det.heavy, det.counts)
            # 2) El semáforo avanza; al iniciar cada ciclo la MLP decide los verdes.
            state = signal.update(t, histories[0].current(), histories[1].current())
            greens = state.decision.greens if state.decision else (None, None)
            # 3) Se dibuja todo.
            for i, lane in enumerate('AB'):
                det, (second, _) = detections[i], frames[i]
                clock = f't = {second:.1f} s · ' if not sources[i].is_image else ''
                slots[f'cam{lane}'].image(simulador.to_jpeg(det.image, max_width=720, quality=75),
                                          caption=f'EN VIVO · {clock}{det.total} vehículos',
                                          width='stretch')
                slots[f'chips{lane}'].markdown(category_chips(det.counts), unsafe_allow_html=True)
                color = state.a if lane == 'A' else state.b
                left = state.remaining_a if lane == 'A' else state.remaining_b
                slots[f'tl{lane}'].markdown(signal_html(f'Avenida {lane}', color, left, greens[i], det.total),
                                            unsafe_allow_html=True)
            if state.decision:
                d = state.decision
                slots['decision'].markdown(
                    f'<div class="live-decision"><b>Decisión de la red</b> · '
                    f'YOLO veía A = {d.detected[0].total} veh. ({d.detected[0].heavy} pesados) y '
                    f'B = {d.detected[1].total} veh. ({d.detected[1].heavy} pesados) → '
                    f'<span class="g">verde A {d.greens[0]} s</span> · <span class="g">verde B {d.greens[1]} s</span> · '
                    f'ciclo {d.cycle_seconds} s{" · conteo saturado al rango entrenado" if any(d.saturated) else ""}</div>',
                    unsafe_allow_html=True)
            else:
                slots['decision'].markdown('<div class="live-decision">Despeje inicial: ambas avenidas en rojo '
                                           f'durante 2 s. La MLP decidirá en {max(0.0, 2 - t):.0f} s…</div>',
                                           unsafe_allow_html=True)
            fps_now = 1 / max(1e-3, monotonic() - tick)
            slots['status'].caption(f'{t:.0f} s en vivo · {min(fps_now, target_fps):.1f} fotogramas por segundo')
            sleep(max(0.0, 1 / target_fps - (monotonic() - tick)))
    finally:
        for source in sources:
            source.close()


with tab_live:
    live_types = PHOTO_TYPES + VIDEO_TYPES
    live_suffixes = tuple(f'.{t}' for t in live_types)
    examples = example_files('videos', live_suffixes) + example_files('fotos', live_suffixes)
    origin = st.radio('Origen en vivo', ['Usar ejemplos incluidos', 'Subir mis archivos'], horizontal=True,
                      key='live_origin', label_visibility='collapsed')
    pick_a, pick_b = st.columns(2)
    live_paths = []
    for column, lane in ((pick_a, 'A'), (pick_b, 'B')):
        with column:
            if origin == 'Subir mis archivos':
                upload = st.file_uploader(f'Cámara · Avenida {lane} (foto o video)', type=live_types,
                                          key=f'live_up_{lane}')
                live_paths.append(live_file(lane, upload) if upload else None)
            else:
                default = next((i for i, p in enumerate(examples) if f'_{lane}_' in p.name and p.suffix == '.mp4'), 0)
                live_paths.append(st.selectbox(f'Cámara · Avenida {lane}', examples, index=default,
                                               format_func=lambda p: ('Video: ' if p.suffix == '.mp4' else 'Foto: ') + nice(p),
                                               key=f'live_ex_{lane}'))
    b1, b2, b3 = st.columns([1.2, 1, 2.2])
    start_live = b1.button('Iniciar', type='primary', disabled=not models_ready, width='stretch')
    b2.button('Detener', width='stretch')
    target_fps = b3.select_slider('Fotogramas analizados por segundo', [1, 2, 3, 4, 5, 6, 8, 10], value=5,
                                  key='live_fps', help='Más fotogramas = video más fluido, pero más uso de CPU.')

    slots = {'status': st.empty(), 'decision': st.empty()}
    # Tablero: [cámara A] [semáforo A] [semáforo B] [cámara B]
    cam_a, light_a, light_b, cam_b = st.columns([3, 1.15, 1.15, 3])
    for column, light, lane in ((cam_a, light_a, 'A'), (cam_b, light_b, 'B')):
        with column:
            slots[f'cam{lane}'] = st.empty()
            slots[f'chips{lane}'] = st.empty()
        slots[f'tl{lane}'] = light.empty()
    for lane in 'AB':  # estado en reposo
        slots[f'tl{lane}'].markdown(signal_html(f'Avenida {lane}', 'Rojo', 0, None, 0), unsafe_allow_html=True)
        slots[f'cam{lane}'].markdown(f'<div class="cam-off">Cámara Avenida {lane}</div>',
                                     unsafe_allow_html=True)
    slots['decision'].markdown('<div class="live-decision">Elige una foto o video por avenida y pulsa <b>Iniciar</b>.</div>',
                               unsafe_allow_html=True)

    if start_live:
        if None in live_paths:
            st.warning('Falta una foto o video: elige un archivo para cada avenida.')
        else:
            try:
                run_live(live_paths, float(target_fps), slots)
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f'No se pudo reproducir la cámara. {error}')
