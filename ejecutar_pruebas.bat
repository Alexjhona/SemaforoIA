@echo off
REM Ejecuta el diagnostico y las pruebas automaticas. Resultado en pruebas_windows.log
cd /d "%~dp0"
set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo Ejecutando diagnostico y pruebas... (resultado en pruebas_windows.log)
"%PY%" diagnostico.py > pruebas_windows.log 2>&1
"%PY%" -m unittest discover -s tests -v >> pruebas_windows.log 2>&1
echo FIN >> pruebas_windows.log
type pruebas_windows.log
timeout /t 8
