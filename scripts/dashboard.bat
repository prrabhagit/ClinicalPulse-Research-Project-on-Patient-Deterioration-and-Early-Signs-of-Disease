@echo off
cd /d "%~dp0.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%\src;%CD%;%PYTHONPATH%
if "%CKPT%"=="" set CKPT=runs\clinicalpulse-cxr\best.pt
set CLINICALPULSE_CKPT=%CKPT%
python -m uvicorn app.server:app --port 8000
