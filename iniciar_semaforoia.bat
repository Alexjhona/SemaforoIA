@echo off
REM Inicia SemaforoIA en el navegador (http://localhost:8501) usando la CPU.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
) else (
    python -m streamlit run app.py --server.port 8501
)
pause
