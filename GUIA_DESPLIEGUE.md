# SemaforoIA · Guía de despliegue

Semáforo inteligente para un cruce de dos avenidas: **YOLOv8n** detecta y cuenta los vehículos y una **red neuronal MLP** decide los segundos de verde de cada avenida. La aplicación web está hecha con Streamlit y funciona solo con CPU (no necesita tarjeta gráfica).

Repositorio: https://github.com/Alexjhona/SemaforoIA

---

## 1. Contenido del ZIP

| Carpeta / archivo | Qué es |
|---|---|
| `app.py` | Aplicación web (punto de entrada). |
| `models/best.pt` | Modelo YOLOv8n entrenado (detección de vehículos). |
| `models/controlador_tiempos.pt` | Red neuronal MLP entrenada (segundos de verde). |
| `detector.py`, `analisis.py`, `en_vivo.py`, `video.py`, `simulador.py`, `modelos.py`, `controller.py`, `traffic.py` | Módulos del sistema. |
| `interfaz/`, `style.css`, `.streamlit/config.toml` | Diseño de la página. |
| `ejemplos/` | Fotos y videos de prueba. |
| `resultados/`, `.metricas_colab.json` | Métricas y gráficas del entrenamiento en Colab. |
| `Entrenamiento_SemaforoIA.ipynb`, `sumo_experiment.py`, `train_controller.py` | Entrenamiento (solo en Colab, no se ejecuta en la laptop). |
| `tests/`, `diagnostico.py`, `ejecutar_pruebas.bat` | Pruebas automáticas y diagnóstico del equipo. |
| `requirements.txt`, `packages.txt` | Dependencias de Python y del sistema (para la nube). |
| `GUIA_EXPOSICION.md` | Guía para sustentar el proyecto. |

Los modelos ya están entrenados: **no hay que volver a entrenar nada**.

---

## 2. Opción A · Ejecutar en una laptop con Windows

Requisitos: Python 3.12, 3.13 o 3.14 (https://www.python.org/downloads/). Al instalarlo, marcar **«Add python.exe to PATH»**.

1. Descomprimir el ZIP, por ejemplo en `D:\SemaforoIA`.
2. Abrir PowerShell en esa carpeta (en el Explorador: clic derecho en un espacio vacío → «Abrir en Terminal»).
3. Crear el entorno virtual e instalar las dependencias (solo la primera vez, tarda unos minutos):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Iniciar la aplicación:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

   o hacer doble clic en **`iniciar_semaforoia.bat`**.

5. Abrir **http://localhost:8501** en el navegador.

Para comprobar que todo está bien: doble clic en **`ejecutar_pruebas.bat`**. El resultado queda en `pruebas_windows.log` y debe terminar en `OK`.

---

## 3. Opción B · Subir el proyecto a GitHub (desde la web)

1. Entrar a https://github.com/new, escribir el nombre del repositorio (por ejemplo `SemaforoIA`), elegir **Public** y pulsar **Create repository**.
2. En la página del repositorio vacío, pulsar **«uploading an existing file»**.
3. Arrastrar **todo el contenido** de la carpeta descomprimida (no la carpeta `.venv` si existe).
4. Escribir un mensaje (por ejemplo «Primera versión») y pulsar **Commit changes**.

La web de GitHub acepta archivos de hasta 25 MB; el modelo más grande (`best.pt`) pesa 6.5 MB, así que se sube sin problema. Las carpetas que empiezan con punto (`.streamlit`) a veces no se arrastran en Windows: si falta, crearla con **Add file → Create new file** escribiendo `.streamlit/config.toml` y copiando su contenido.

---

## 4. Opción C · Publicar gratis en Streamlit Community Cloud

Necesita el proyecto en GitHub (opción B o el repositorio indicado arriba).

1. Entrar a **https://share.streamlit.io** e iniciar sesión con la cuenta de GitHub.
2. Pulsar **Create app** → **Deploy a public app from GitHub**.
3. Completar:
   - Repository: `Alexjhona/SemaforoIA`
   - Branch: `main`
   - Main file path: `app.py`
4. Abrir **Advanced settings**, elegir **Python 3.12** y guardar.
5. Pulsar **Deploy**. La primera vez tarda varios minutos (instala PyTorch, YOLO y OpenCV).
6. Al terminar, la app tiene un enlace público del tipo `https://semaforoia.streamlit.app`.

Notas sobre la nube:

- `requirements.txt` instala PyTorch **solo para CPU** (más liviano) y `packages.txt` instala las librerías del sistema que necesita OpenCV.
- El CPU de la nube es compartido: las fotos y videos funcionan bien, pero el modo **En vivo** va más lento que en la laptop.
- Si nadie abre la app durante varias horas, Streamlit la «duerme»; basta con abrir el enlace y pulsar el botón para despertarla (tarda alrededor de un minuto).
- Render (plan gratis) **no se recomienda**: 512 MB de RAM y 0.1 CPU no alcanzan para PyTorch y YOLO.

---

## 5. Uso rápido de la aplicación

1. En la barra izquierda, los dos modelos deben tener el **punto verde**.
2. Pestaña **En vivo**: dejar los videos de ejemplo (o subir una foto o video por avenida) y pulsar **Iniciar**. Las cámaras muestran los cuadros de YOLO y los semáforos cambian de luz con su contador; arriba aparece la decisión de la red.
3. Pestañas **Fotos** y **Videos**: análisis de una foto o video por avenida con la simulación de semáforos.
4. Pestaña **Probar la red**: escribir conteos y ver cuántos segundos de verde decide la MLP.

---

## 6. Problemas comunes

| Problema | Solución |
|---|---|
| `python` no se reconoce | Reinstalar Python marcando «Add python.exe to PATH». |
| `No module named ...` | Repetir `pip install -r requirements.txt` con el Python del `.venv`. |
| El puerto 8501 está ocupado | Ejecutar `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` y abrir http://localhost:8502. |
| Los modelos aparecen con punto rojo | Verificar que existan `models/best.pt` y `models/controlador_tiempos.pt`. |
| En la nube: «This app has gone over its resource limits» | En el menú de la app, pulsar **Reboot app**. |
| YOLO detecta pocos vehículos | El detector se entrenó con cámaras elevadas de una intersección; funciona mejor con vistas similares. En «Ajustes avanzados» se puede bajar el umbral de confianza a 0.15. |
