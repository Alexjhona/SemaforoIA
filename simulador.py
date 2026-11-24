"""ETAPA 3 · Simulación visual de los dos semáforos.

Python prepara los datos (fotogramas anotados por YOLO y decisiones de la MLP)
y los inserta en una página HTML/JavaScript (`interfaz/simulador.html`). La
animación de luces, temporizadores y botones corre en el navegador, por eso es
fluida y no vuelve a ejecutar los modelos.

La secuencia de fases la genera `traffic.build_cycle` (verde → amarillo 3 s →
rojo total 2 s → verde de la otra avenida); el navegador solo la reproduce.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from analisis import Decision
from traffic import ALL_RED, YELLOW


TEMPLATE = Path(__file__).resolve().parent / 'interfaz' / 'simulador.html'


def to_jpeg(image: Image.Image, max_width: int = 640, quality: int = 72) -> bytes:
    """Comprime una imagen anotada para enviarla al navegador sin gastar memoria."""
    image = image.copy()
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)))
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    return buffer.getvalue()


def _data_uri(jpeg: bytes) -> str:
    return 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')


def lane_payload(name: str, source: str, frames: list[tuple[float, int, bytes]], duration: float) -> dict:
    """frames: lista de (segundo, vehículos detectados, jpeg)."""
    return {'name': name, 'source': source, 'duration': duration,
            'frames': [{'t': t, 'n': n, 'src': _data_uri(jpeg)} for t, n, jpeg in frames]}


def decision_payload(decision: Decision) -> dict:
    lane_a, lane_b = decision.detected
    sat_a, sat_b = decision.saturated
    return {'label': decision.label, 'greens': list(decision.greens), 'phases': decision.phases,
            'cycle': decision.cycle_seconds, 'input': list(decision.mlp_input),
            'detA': lane_a.total, 'detB': lane_b.total, 'heavyA': lane_a.heavy, 'heavyB': lane_b.heavy,
            'satA': sat_a, 'satB': sat_b}


def build_html(lane_a: dict, lane_b: dict, decisions: list[Decision], window: float | None,
               autostart: bool = False) -> str:
    data = {'lanes': {'A': lane_a, 'B': lane_b}, 'decisions': [decision_payload(d) for d in decisions],
            'window': window, 'allRed': ALL_RED, 'yellow': YELLOW, 'autostart': autostart}
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    return TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', payload)


def show(lane_a: dict, lane_b: dict, decisions: list[Decision], window: float | None = None,
         height: int = 870) -> None:
    """Inserta el simulador en la página de Streamlit."""
    import streamlit as st

    html = build_html(lane_a, lane_b, decisions, window)
    if hasattr(st, 'iframe'):  # Streamlit reciente: la altura se ajusta al contenido
        st.iframe(html, height='content')
    else:                      # versiones anteriores
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=False)
