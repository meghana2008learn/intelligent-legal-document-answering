# PowerShell 1-Click Runner for Intelligent Legal Document Answering
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "   ⚖️ Intelligent Legal Document Answering (BTech AI Capstone Project)" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please set up the environment with:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

Write-Host "Launching Streamlit in default browser..." -ForegroundColor Green
& $VenvPython -m streamlit run (Join-Path $PSScriptRoot "app.py")
