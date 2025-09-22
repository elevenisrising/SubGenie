@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Install/repair PyTorch stack and related wheels from local whl folder
REM Prefer path-based env to avoid global writes; fallback to named env

set LOCAL_ENV_PATH=D:\Dream\translate\.conda\asr-env
set CONDA_RUN=conda run -n asr-env
if exist "%LOCAL_ENV_PATH%\python.exe" (
  set CONDA_RUN=conda run -p %LOCAL_ENV_PATH%
  echo Detected local env: %LOCAL_ENV_PATH%
) else (
  echo Using named env: asr-env
)

set WHL_DIR=..\whl
if not exist "%WHL_DIR%" (
  echo [ERROR] Local wheel folder not found: %WHL_DIR%
  echo         Please put your *.whl files under: %CD%\..\whl
  exit /b 1
)

echo [INFO] Using wheel folder: %WHL_DIR%

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Conda not found in PATH. Install Miniconda/Anaconda and reopen terminal.
  exit /b 1
)

echo [STEP] Installing PyTorch stack from local wheels (best match)...
%CONDA_RUN% python -m pip install --no-index --find-links "%WHL_DIR%" torch* torchaudio* torchvision* || (
  echo [WARN] Could not find matching torch/torchaudio wheels or install failed.
)

echo [STEP] Installing spaCy model from local wheels if available...
%CONDA_RUN% python -m pip install --no-index --find-links "%WHL_DIR%" en_core_web_sm* || (
  echo [INFO] No local en_core_web_sm wheel found; skipping.
)

echo [STEP] Installing whisper/whisperx from local wheels if available...
%CONDA_RUN% python -m pip install --no-index --find-links "%WHL_DIR%" openai_whisper* whisperx* || (
  echo [INFO] No local whisper/whisperx wheels found; skipping.
)

echo [STEP] Installing common runtime deps (requests)...
%CONDA_RUN% python -m pip install --no-index --find-links "%WHL_DIR%" requests* || (
  echo [INFO] No local requests wheel; skipping.
)

echo [VERIFY] Torch and torchaudio versions:
%CONDA_RUN% python - <<PY
import torch, sys
print('torch:', torch.__version__)
try:
  import torchaudio
  print('torchaudio:', torchaudio.__version__)
except Exception as e:
  print('torchaudio not available:', e)
PY

echo.
echo [DONE] Local wheel installation step completed.
endlocal
