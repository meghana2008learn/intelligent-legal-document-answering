@echo off
title Intelligent Legal Document Answering
echo ==============================================================================
echo    ⚖️ Intelligent Legal Document Answering (BTech AI Capstone Project)
echo ==============================================================================
echo Activating virtual environment and starting Streamlit application...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please run:
    echo   python -m venv venv
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

call .\venv\Scripts\activate.bat
python -m streamlit run app.py

pause
