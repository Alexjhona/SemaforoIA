"""ETAPA 1 · Detección de vehículos con YOLOv8n (best.pt).

Este módulo recibe una imagen (foto o fotograma de video), ejecuta el detector
entrenado en Colab y devuelve:
  * la imagen con los cuadros delimitadores dibujados,
  * el conteo por categoría (Auto, Motocicleta, Camión, Bus, Bicicleta),
  * el total de vehículos y el número de vehículos pesados (Camión + Bus).

Solo se cuentan los vehículos cuyo centro cae dentro de la "zona de detección"
(el tramo de la avenida que se aproxima al cruce). Por defecto la zona es toda
la imagen.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# Clases del dataset de Roboflow (nombres en inglés) → etiqueta en español.
VEHICLES = {
    'car': 'Auto', 'truck': 'Camión', 'bus': 'Bus',
    'bicycle': 'Bicicleta', 'motorcycle': 'Motocicleta',
}
HEAVY = ('Camión', 'Bus')
CATEGORY_ORDER = ('Auto', 'Motocicleta', 'Camión', 'Bus', 'Bicicleta')
# Colores BGR (OpenCV) para cada categoría.
COLORS = {
    'Auto': (183, 230, 105), 'Motocicleta': (255, 191, 64), 'Camión': (64, 140, 255),
    'Bus': (255, 105, 200), 'Bicicleta': (210, 210, 210),
}
MAX_IMAGE_PIXELS = 24_000_000

# La app trabaja sin conexión: evita que Ultralytics intente instalar paquetes o
# consultar Internet durante la exposición.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')


@dataclass(frozen=True)
class Zone:
    """Zona de detección expresada en fracciones (0 a 1) del ancho y alto de la imagen."""
    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0

    @property
    def is_full(self) -> bool:
        return (self.left, self.top, self.right, self.bottom) == (0.0, 0.0, 1.0, 1.0)

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (round(self.left * width), round(self.top * height),
                round(self.right * width), round(self.bottom * height))


@dataclass(frozen=True)
class DetectionSettings:
    """Parámetros de inferencia ajustables desde la interfaz."""
    confidence: float = 0.25
    image_size: int = 640          # resolución de entrada de YOLO (se entrenó con 640)
    zone: Zone = field(default_factory=Zone)
    agnostic_nms: bool = True      # evita contar el mismo vehículo como Auto y Camión a la vez
    drop_nested: bool = True       # descarta cajas pequeñas encerradas dentro de otra (ruedas, ventanas)
    max_width: int = 960           # ancho máximo de la imagen anotada que se muestra


@dataclass
class Detection:
    """Resultado de analizar una imagen."""
    image: Image.Image                       # imagen anotada (RGB)
    counts: dict[str, int]                   # vehículos por categoría dentro de la zona
    boxes: list[tuple] = field(default_factory=list)  # (x1, y1, x2, y2, etiqueta, confianza, dentro)
    inference_ms: float = 0.0

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def heavy(self) -> int:
        return sum(self.counts.get(name, 0) for name in HEAVY)


def decode_image(blob: bytes) -> Image.Image:
    """Abre una fotografía subida, corrige la orientación EXIF y la pasa a RGB."""
    try:
        with Image.open(BytesIO(blob)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError('Utiliza una imagen de hasta 24 megapíxeles.')
            return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ImportError) as error:
        # ImportError: Ultralytics intenta abrir formatos HEIC con un paquete opcional.
        raise ValueError('No se pudo abrir la imagen. Usa JPG, PNG o WEBP.') from error


def _nested(box, others) -> bool:
    """True si al menos el 85 % de la caja está dentro de otra caja claramente mayor."""
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for ox1, oy1, ox2, oy2 in others:
        other_area = (ox2 - ox1) * (oy2 - oy1)
        if other_area <= area * 1.5:
            continue
        inter = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        if inter / area >= 0.85:
            return True
    return False


def _draw(frame_rgb: np.ndarray, boxes: list[tuple], zone: Zone, max_width: int) -> Image.Image:
    """Dibuja la zona de detección y los cuadros delimitadores con etiquetas en español."""
    import cv2

    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    if not zone.is_full:
        # Oscurece lo que queda fuera de la zona para que se vea qué tramo se cuenta.
        x1, y1, x2, y2 = zone.pixels(width, height)
        shade = (canvas * 0.45).astype(np.uint8)
        shade[y1:y2, x1:x2] = canvas[y1:y2, x1:x2]
        canvas = shade
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (105, 230, 183), 2)
    thickness = max(1, round(width / 640))
    scale = max(0.4, width / 1400)
    for bx1, by1, bx2, by2, label, conf, inside in boxes:
        color = COLORS.get(label, (255, 255, 255)) if inside else (120, 120, 120)
        p1, p2 = (int(bx1), int(by1)), (int(bx2), int(by2))
        cv2.rectangle(canvas, p1, p2, color, thickness + (1 if inside else 0))
        if inside:
            text = f'{label} {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            ty = max(th + 4, p1[1])
            cv2.rectangle(canvas, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), color, -1)
            cv2.putText(canvas, text, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (15, 20, 25), 1, cv2.LINE_AA)
    if width > max_width:
        canvas = cv2.resize(canvas, (max_width, round(height * max_width / width)),
                            interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


class VehicleDetector:
    """Envuelve el modelo YOLOv8n entrenado (best.pt). Se carga una sola vez."""

    def __init__(self, weights: Path):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task='detect')
        # Índice de clase del modelo → etiqueta en español (solo clases vehiculares).
        self.labels = {index: VEHICLES[name.strip().lower()] for index, name in self.model.names.items()
                       if name.strip().lower() in VEHICLES}
        if not self.labels:
            raise ValueError('El modelo no incluye las clases vehiculares del dataset.')

    def warm_up(self) -> None:
        """Primera inferencia en vacío: la siguiente (la del usuario) ya es rápida."""
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)

    def predict(self, images: list, confidence: float = 0.25,
                settings: DetectionSettings | None = None) -> list[Detection]:
        """Analiza una lista de imágenes PIL o arreglos RGB y devuelve un Detection por imagen."""
        settings = settings or DetectionSettings(confidence=confidence)
        arrays = [np.asarray(image.convert('RGB')) if isinstance(image, Image.Image) else image
                  for image in images]
        results = self.model.predict(
            arrays, classes=list(self.labels), imgsz=settings.image_size, conf=settings.confidence,
            agnostic_nms=settings.agnostic_nms, max_det=300, verbose=False, device='cpu')
        return [self._summarize(frame, result, settings) for frame, result in zip(arrays, results)]

    def _summarize(self, frame: np.ndarray, result, settings: DetectionSettings) -> Detection:
        height, width = frame.shape[:2]
        xyxy = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        scores = result.boxes.conf.tolist()
        zx1, zy1, zx2, zy2 = settings.zone.pixels(width, height)
        boxes = []
        for index, (box, cls, score) in enumerate(zip(xyxy, classes, scores)):
            if int(cls) not in self.labels:
                continue
            if settings.drop_nested and _nested(box, xyxy[:index] + xyxy[index + 1:]):
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            inside = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
            boxes.append((*box, self.labels[int(cls)], float(score), inside))
        counts = Counter(label for *_, label, _, inside in boxes if inside)
        image = _draw(frame, boxes, settings.zone, settings.max_width)
        speed = getattr(result, 'speed', None) or {}
        return Detection(image, dict(counts), boxes, float(sum(speed.values())))
