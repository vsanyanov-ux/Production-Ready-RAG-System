@echo off
echo Installing dependencies for Corporate Bot...
echo.

echo Creating virtual environment (venv)...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing libraries from requirements.txt...
pip install -r requirements.txt

echo.
echo Installation complete!
pause
