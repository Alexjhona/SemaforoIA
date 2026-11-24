# SemaforoIA · Guía para la exposición

## 1. Idea en una frase

Una cámara observa cada avenida; **YOLOv8n** cuenta los vehículos y una **red neuronal MLP** decide cuántos segundos de verde recibe cada avenida. Un controlador de fases seguro convierte esa decisión en el ciclo de dos semáforos.

```
Foto / video ─► YOLOv8n (best.pt) ─► conteo por avenida ─► MLP (controlador_tiempos.pt) ─► segundos de verde ─► semáforos
```

## 2. Cómo iniciar la demostración

1. Doble clic en `iniciar_semaforoia.bat` y abrir **http://localhost:8501**.
2. En la barra izquierda, los dos modelos deben tener el punto verde.
3. Pestaña **En vivo**: dejar los videos de ejemplo (o subir una foto o video por avenida) y pulsar **Iniciar**.
   - Las cámaras muestran los cuadros de YOLO sobre cada vehículo, en tiempo real.
   - Los dos semáforos cambian de luz con su contador.
   - Arriba aparece la decisión de la red: cuántos vehículos vio y cuántos segundos de verde dio a cada avenida. Se renueva al comenzar cada ciclo.
   - **Detener** apaga las cámaras.
4. Opcional: pestaña **Fotos** (análisis de una foto por avenida) o **Probar la red** (escribir conteos y ver la respuesta de la red).

## 3. YOLOv8n: detección de vehículos (`detector.py`)

- **Qué es:** red convolucional de detección de objetos «de una sola pasada» (*You Only Look Once*). La versión *n* (nano) tiene unos 3 millones de parámetros y por eso funciona en CPU.
- **Cómo funciona:** la imagen se redimensiona a 640 × 640 px. Una red troncal (*backbone*) extrae características, un «cuello» (*neck*) combina varias escalas para ver vehículos grandes y pequeños, y la «cabeza» predice para cada zona una caja (x, y, ancho, alto), la clase y la confianza. La supresión de no máximos (**NMS**) elimina cajas repetidas sobre el mismo vehículo.
- **Entrenamiento (Colab, GPU T4):** dataset *Intersection-traffic-detection* de Roboflow, 6.703 imágenes, 30 épocas, 5 clases: `car`, `motorcycle`, `truck`, `bus`, `bicycle`.
- **Resultados en la prueba oficial (671 imágenes):** precisión 55,4 %, sensibilidad 57,5 %, **mAP50 61,6 %**, mAP50-95 50,2 %. Por clase, el mejor es *car* (mAP50 92 %); *bicycle* casi no tenía ejemplos (8 %).
- **En la app:** se cuentan solo las clases vehiculares; Camión + Bus = **pesados**. Ajustes:
  - *Umbral de confianza*: menos umbral = más detecciones, pero más falsos positivos.
  - *Una sola etiqueta por vehículo* (NMS agnóstico): evita contar el mismo vehículo como Auto y como Camión.
  - *Ignorar cajas dentro de otra*: descarta ruedas o ventanas detectadas dentro de un vehículo.
  - *Zona de detección*: solo cuenta los vehículos cuyo centro está en el tramo que llega al cruce.

## 4. La MLP: segundos de verde (`controller.py`, `analisis.py`)

- **Entradas (6):** vehículos A, vehículos B, pesados A, pesados B, demanda total y proporción de A. Se normalizan con la media y desviación del entrenamiento.
- **Arquitectura:** 6 → 64 → 32 → 2 neuronas, activación ReLU. La salida pasa por una sigmoide escalada a **10–35 s**, así nunca produce tiempos imposibles.
- **De dónde aprendió:** 192 escenarios de tráfico simulados en **Eclipse SUMO**. En cada escenario se probaron todas las combinaciones de verde (10, 15, …, 35 s para cada avenida) y la etiqueta fue la combinación con **menor espera** de los vehículos.
- **Entrenamiento:** 120 épocas, AdamW, pérdida Smooth L1; se guardó la mejor época por validación (época 67).
- **Error en prueba (29 escenarios):** MAE 4,66 s en la avenida A y 6,55 s en la B.
- **Rango entrenado:** hasta 23 vehículos en A y 25 en B. Si YOLO cuenta más, la app **satura** el conteo al máximo entrenado (se avisa en pantalla) en lugar de pedirle a la red algo que nunca vio.
- **Para comprobar que la decisión es de la red:** la pestaña de prueba directa y la tabla de decisiones muestran exactamente qué entrada recibió la MLP y qué devolvió.

## 5. Análisis de video (`video.py`, `analisis.py`)

1. Se leen solo *N* fotogramas por segundo (por defecto 2); el resto se salta sin decodificar, ahorrando CPU y memoria.
2. YOLO analiza cada fotograma muestreado.
3. Los conteos se agrupan en **ventanas de decisión** (por defecto 5 s). En cada ventana se usa la **mediana**, robusta a detecciones momentáneas.
4. La MLP predice un nuevo par de verdes por ventana: el semáforo **se adapta** cuando cambia el tráfico.
5. En la simulación, al comenzar cada ciclo se toma la decisión de la ventana de video correspondiente.

## 6. Simulación segura de los semáforos (`traffic.py`, `simulador.py`)

Secuencia de cada ciclo:

| Fase | Avenida A | Avenida B | Duración |
|---|---|---|---|
| 1 | Verde | Rojo | verde A (MLP) |
| 2 | Amarillo | Rojo | 3 s |
| 3 | Rojo | Rojo | 2 s (despeje) |
| 4 | Rojo | Verde | verde B (MLP) |
| 5 | Rojo | Amarillo | 3 s |
| 6 | Rojo | Rojo | 2 s (despeje) |

- **Nunca hay dos verdes a la vez**; esto lo comprueban las pruebas automáticas.
- Sin vehículos detectados: ambas avenidas permanecen en rojo.
- La red solo decide **cuánto** dura cada verde; el **orden** seguro lo fija el programa.
- La animación corre en el navegador (JavaScript): botones Iniciar, Detener, Reiniciar y velocidad 1×/2×/4×.

## 7. Optimización para la laptop (CPU)

- Los modelos se cargan **una sola vez** (`st.cache_resource`) y se hace una inferencia de calentamiento.
- Sin GPU: `device='cpu'`, hilos limitados para no congelar el equipo.
- Muestreo de fotogramas, duración máxima analizada y fotogramas reducidos a 960 px como máximo.
- Las imágenes anotadas se guardan como JPEG pequeños; los videos subidos se borran al terminar.
- Resolución de inferencia ajustable (640, 512, 416 o 320 px). En la laptop de la exposición (CPU, 640 px) cada foto tarda unos 40 ms y un video de 12 s se analiza en 1 a 7 s (medido con `diagnostico.py`).

## 8. Limitaciones (decirlas suma puntos)

- El detector se entrenó con cámaras **elevadas** de una intersección. En escenas distintas (primeros planos, otro país, otro ángulo) detecta solo parte de los vehículos; por ejemplo, las motocicletas de los videos de ejemplo casi no se detectan.
- La MLP aprendió de un cruce **simulado** de un carril por avenida; una foto mide la demanda visible, no la velocidad ni el flujo.
- Es una simulación académica, no un sistema para controlar semáforos reales.

## 9. Comprobar el equipo antes de exponer

Doble clic en `ejecutar_pruebas.bat`: muestra las versiones, carga los dos modelos, mide los tiempos y ejecuta las 15 pruebas automáticas. El resultado queda en `pruebas_windows.log`.

## 10. Preguntas probables

- **¿Por qué dos redes y no una?** Cada una resuelve una tarea distinta: visión (YOLO) y decisión (MLP). Así se puede evaluar y mejorar cada parte por separado.
- **¿Por qué una MLP y no una regla fija?** En SUMO, la MLP aprendió combinaciones que reducen la espera mejor que un tiempo fijo; además considera los vehículos pesados.
- **¿Qué pasa si YOLO se equivoca?** El error de conteo se propaga a la MLP; por eso se usa la mediana en video, el umbral de confianza y la zona de detección.
- **¿Se entrenó algo en la laptop?** No. Los pesos se entrenaron en Colab y la app solo los lee.
