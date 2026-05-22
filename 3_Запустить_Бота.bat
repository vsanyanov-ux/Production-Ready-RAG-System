@echo off
echo Starting Corporate Bot...
echo.

echo Checking if Ollama is running...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not running! Please start Ollama before using the bot.
    pause
    exit /b
)

echo Activating environment...
call venv\Scripts\activate

echo Starting Streamlit interface...
python run_app.py
pause
