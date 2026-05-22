@echo off
echo Starting Corporate Document Ingestion...
echo.

echo Activating virtual environment...
call venv\Scripts\activate

echo Checking data directory...
if not exist data (
    mkdir data
    echo Data directory created! Please place your documents ^(txt, pdf, docx^) there and run this script again.
    pause
    exit /b
)

echo.
echo Running ingestion process...
python ingest.py

echo.
pause
