# SemaforoIA

Proyecto universitario de detección de vehículos y predicción de segundos de luz verde. En Colab se entrenan YOLOv8n con imágenes etiquetadas y un perceptrón multicapa con escenarios de Eclipse SUMO. Streamlit conecta los dos modelos para analizar imágenes y simular el ciclo de un cruce.

## Entrenamiento en Colab

1. Abre `Entrenamiento_SemaforoIA.ipynb` y elige **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4** si aún no está seleccionada.
2. Ejecuta las primeras celdas y autoriza el montaje de Google Drive cuando Colab lo solicite.
3. Ejecuta los bloques 3 al 5 para generar los escenarios, entrenar la MLP y evaluar los tiempos con SUMO. El bloque 6 descarga `controlador_tiempos.pt`.
4. Para importar [Intersection-traffic-detection v1](https://universe.roboflow.com/remas-workspace/intersection-traffic-detection-sf6fy/dataset/1), habilita el secreto `ROBOFLOW_API_KEY` para este notebook o guarda la exportación YOLOv8 como `Mi unidad/SemaforoIA/dataset_semaforoia_yolov8.zip`.
5. Ejecuta los bloques 7 al 12 para entrenar YOLOv8n. El modelo principal usa 30 épocas y la validación cruzada usa tres pliegues de 10 épocas. La prueba final usa el `best.pt` del entrenamiento principal. Los bloques 13 y 14 descargan el detector y el dataset.

Si Colab se desconecta, ejecuta de nuevo desde el primer bloque y conserva el mismo `RUN_ID` para recuperar el detector. La etapa SUMO reutiliza los escenarios terminados guardados en Drive. La MLP se ajusta de nuevo en pocos pasos a partir de dichos datos.

Drive conserva los datos bajo `SemaforoIA/dataset`, los modelos bajo `SemaforoIA/modelos`, y las gráficas, matrices, checkpoints y métricas bajo `SemaforoIA/resultados`. `SemaforoIA/resultados/controlador_v1` contiene los datos de SUMO, las particiones, la curva de aprendizaje y las comparaciones del controlador. `SemaforoIA/auditoria` contiene el inventario del dataset de imágenes.

La validación cruzada conserva las secuencias completas. Los nombres de la exportación mantienen cámara, hora y número de fotograma. Las 10 secuencias identificadas aparecen en las tres particiones oficiales. La prueba oficial evalúa imágenes reservadas y la validación cruzada evalúa secuencias reservadas de la misma cámara. Si se dispone de identificadores de origen adicionales, `grupos_origen.csv` puede agregarse a `SemaforoIA` en Drive con columnas `imagen` y `grupo`.

## Aplicación web (Streamlit, CPU)

La aplicación usa los dos modelos ya entrenados y **no entrena nada** en la laptop. Funciona en Windows solo con CPU.

```powershell
cd "D:\Proyectos Grandes Curos 2026-2\Inteligencia Artificial\Sesion 3 con Ciencia de datos\SemaforoIA"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

También se puede hacer doble clic en `iniciar_semaforoia.bat`. La aplicación se abre en **http://localhost:8501**.

Si se instala en otro equipo: `python -m pip install -r requirements.txt` (Python 3.12 a 3.14; instala PyTorch solo para CPU).

Pestañas:

- **Fotografías**: una foto por avenida → YOLOv8n dibuja los cuadros y cuenta por categoría → la MLP da los segundos de verde → simulación.
- **Videos**: un MP4 corto por avenida. Se analizan *N* fotogramas por segundo (ajustable) y la MLP decide de nuevo en cada ventana de video.
- **Prueba directa de la MLP**: conteos escritos a mano (indicado en pantalla) para explicar cómo responde la red.
- **Cómo funciona**: etapas y métricas reales de Colab.

La barra lateral muestra el estado de los dos modelos y permite ajustar la confianza, la resolución de inferencia, la zona de detección, la frecuencia de fotogramas, la ventana de decisión y la saturación de conteos. `ejemplos/` contiene fotos y videos de prueba (ver `ejemplos/FUENTES.txt`).

Los verdes quedan entre 10 y 35 segundos en cada vía, con 3 segundos de amarillo y 2 segundos de rojo en ambas vías antes de cambiar el paso. Sin vehículos se mantiene el rojo en las dos vías. Se trata de una simulación académica. Los datos de aprendizaje del controlador proceden de un cruce idealizado y las imágenes representan demanda visible, sin medir velocidad ni flujo por unidad de tiempo.

La guía para la sustentación está en `GUIA_EXPOSICION.md`.

## Publicar gratis en Streamlit Community Cloud

1. Entrar a https://share.streamlit.io e iniciar sesión con la cuenta de GitHub.
2. **Create app** → *Deploy a public app from GitHub* → repositorio `Alexjhona/SemaforoIA`, rama `main`, archivo `app.py`.
3. En **Advanced settings** elegir Python 3.12 y pulsar **Deploy**. La primera instalación tarda varios minutos.

`requirements.txt` instala PyTorch para CPU y `packages.txt` las librerías del sistema que necesita OpenCV. En la nube el CPU es compartido, por lo que el modo En vivo va más lento que en la laptop.

## Archivos principales

- `app.py` · interfaz Streamlit (orquesta las etapas).
- `modelos.py` · carga única de `best.pt` y `controlador_tiempos.pt` y su estado.
- `detector.py` · YOLOv8n: cuadros delimitadores, conteo por categoría y zona de detección.
- `video.py` · lectura de MP4 con muestreo de fotogramas para CPU.
- `analisis.py` · conteos → entradas de la MLP → segundos de verde → plan de fases.
- `controller.py` · definición y carga de la MLP.
- `traffic.py` · fases seguras del cruce (verde, amarillo, rojo total).
- `simulador.py` + `interfaz/simulador.html` · semáforos animados con Iniciar, Detener y Reiniciar.
- `sumo_experiment.py`, `train_controller.py`, `Entrenamiento_SemaforoIA.ipynb` · entrenamiento en Colab (no se ejecutan en la laptop).
- `tests/test_semaforo.py` · pruebas automáticas, incluidas pruebas con los modelos reales.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Datos

Remas Workspace (2026). *Intersection-traffic-detection*, versión 1. Roboflow Universe. Licencia CC BY 4.0. [Ficha del dataset](https://universe.roboflow.com/remas-workspace/intersection-traffic-detection-sf6fy/dataset/1).
