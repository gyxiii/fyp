@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting MusicMotion Studio...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

start "MusicMotion Studio" ".venv\Scripts\python.exe" -m streamlit run "app.py" ^
    --server.port 8501 --server.address 127.0.0.1 --server.headless true

echo Waiting for the server...
timeout /t 6 >nul
start "" "http://127.0.0.1:8501"
echo.
echo MusicMotion Studio is running at http://127.0.0.1:8501
echo Close the "MusicMotion Studio" console window to stop it.
