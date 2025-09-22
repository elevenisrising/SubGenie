@echo off
REM =================================
REM  SubGenie GUI Launcher (Conda)
REM =================================

REM Prefer path-based env to avoid global writes; fallback to named env
set LOCAL_ENV_PATH=D:\Dream\translate\.conda\asr-env
set CONDA_RUN=conda run -n asr-env
if exist "%LOCAL_ENV_PATH%\python.exe" (
  set CONDA_RUN=conda run -p %LOCAL_ENV_PATH%
  echo Detected local env: %LOCAL_ENV_PATH%
) else (
  echo Using named env: asr-env
)

REM Ensure FFmpeg DLLs from Conda env are found first
set PATH=%LOCAL_ENV_PATH%\Library\bin;%PATH%

set TORCHAUDIO_USE_FFMPEG=0
echo Starting SubGenie GUI...
echo This may take a moment...
echo.

REM Use 'conda run' to execute the python script directly in the specified environment.
REM This is the recommended and more reliable way to run a single task in Conda.
%CONDA_RUN% python main_gui.py

echo.
echo Application has been closed. Press any key to exit.
pause >nul
