@echo off
REM scripts\train.bat           -> DenseNet121 with torchvision weights (downloads once)
REM scripts\train.bat offline   -> ResNet18 with bundled weights (no download)
REM Set EPOCHS / BATCH / WORKERS / OUT env vars to override.
cd /d "%~dp0.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%\src;%PYTHONPATH%
if "%EPOCHS%"=="" set EPOCHS=15
if "%BATCH%"=="" set BATCH=32
if "%WORKERS%"=="" set WORKERS=4
if "%OUT%"=="" set OUT=runs\clinicalpulse-cxr
set ARCHARGS=--arch densenet121
if /I "%1"=="offline" set ARCHARGS=--arch resnet18 --init-weights weights\resnet18_a1_0-d63eafa0.pth
python -m clinicalpulse.imaging.train_cxr train --dataset chestxray14 --root data\cxr14 --shards data\shards %ARCHARGS% --epochs %EPOCHS% --warmup 2 --patience 5 --batch-size %BATCH% --workers %WORKERS% --out %OUT%
