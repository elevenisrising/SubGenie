@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ==============================================
echo  SubGenie ASR Environment (manual setup)
echo ==============================================
echo.

REM Manual, step-by-step install without environment YAML.
REM GPU supported. Falls back to local wheels when available.

set LOCAL_ENV_PATH=D:\Dream\translate\.conda\asr-env
set CONDA_RUN=conda run -p %LOCAL_ENV_PATH%

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Conda not found in PATH. Please install Miniconda/Anaconda and reopen the terminal.
  exit /b 1
)

if not exist "%LOCAL_ENV_PATH%\python.exe" (
  echo [STEP] Creating prefix env at: %LOCAL_ENV_PATH%
  conda create -y -p %LOCAL_ENV_PATH% python=3.10 || (
    echo [ERROR] Failed to create conda env at prefix.
    exit /b 1
  )
) else (
  echo [INFO] Using existing env at: %LOCAL_ENV_PATH%
)

echo.
echo [STEP] Ensuring ffmpeg (conda-forge) is installed...
conda install -y -p %LOCAL_ENV_PATH% -c conda-forge ffmpeg || echo [WARN] ffmpeg install failed; continuing.

echo.
echo [STEP] Upgrading pip/setuptools/wheel...
%CONDA_RUN% python -m pip install -U pip setuptools wheel

echo.
echo [STEP] Install base runtime deps (requests, pillow, customtkinter, pydub, srt, tqdm, numpy)...
%CONDA_RUN% python -m pip install requests pillow customtkinter pydub srt tqdm numpy

echo.
echo [STEP] Install audio/processing deps (soundfile)...
%CONDA_RUN% python -m pip install soundfile

echo.
echo [STEP] Install PyTorch stack (prefer local wheels if present)...
set WHL_DIR=%~dp0..\whl
if exist "%WHL_DIR%\torch-*.whl" (
  echo [INFO] Found local wheels in: %WHL_DIR%
  %CONDA_RUN% python -m pip install --no-index --find-links "%WHL_DIR%" torch* torchaudio* torchvision* || echo [WARN] Local torch wheel install failed.
) else (
  echo [INFO] No local wheels found. Attempting online install (CUDA 12.1 default)...
  echo        If this fails or CUDA mismatch occurs, run scripts\use_local_whl.bat
  %CONDA_RUN% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || echo [WARN] Torch install (cu121) failed.
)

echo.
echo [STEP] Install ASR/translation stack (whisper, whisperx, faster-whisper, ctranslate2, spacy)...
%CONDA_RUN% python -m pip install openai-whisper whisperx faster-whisper ctranslate2 spacy

echo.
echo [STEP] Download spaCy model (best-effort)...
%CONDA_RUN% python -m spacy download en_core_web_sm || echo [WARN] spaCy model download failed.

echo.
echo [VERIFY] Quick import check...
%CONDA_RUN% python - <<PY
import sys
print('python =', sys.version)
try:
    import torch; print('torch =', torch.__version__, 'cuda =', torch.version.cuda, 'cuda_available =', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu =', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch check failed:', e)
try:
    import whisperx, whisper, faster_whisper, ctranslate2
    import spacy, srt, requests, PIL, customtkinter, pydub, soundfile, librosa
    print('imports ok')
except Exception as e:
    print('import error:', e)
PY

echo.
echo [DONE] Manual setup finished.
echo        - Start GUI: start_gui.bat
echo        - CLI run:   scripts\run_asr_cli.bat <input> [args]
echo        - Fix torch via local wheels: scripts\use_local_whl.bat
echo.
endlocal
