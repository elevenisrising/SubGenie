@echo off
setlocal

set LOCAL_ENV_PATH=D:\Dream\translate\.conda\asr-env
set CONDA_RUN=conda run -n asr-env
if exist "%LOCAL_ENV_PATH%\python.exe" (
  set CONDA_RUN=conda run -p %LOCAL_ENV_PATH%
  echo Detected local env: %LOCAL_ENV_PATH%
) else (
  echo Using named env: asr-env
)

set SCRIPT=src\processing\main.py

if "%~1"=="" (
  echo Usage: %~nx0 ^<input_audio_or_project^> [extra args]
  echo Example: %~nx0 input_audio\demo.mp4 --model large-v3 --target-language zh-CN
  exit /b 1
)

REM Ensure FFmpeg DLLs from Conda env are found first
set PATH=%LOCAL_ENV_PATH%\Library\bin;%PATH%

REM Disable torchaudio FFmpeg extension to avoid DLL issues; use soundfile backend where possible
set TORCHAUDIO_USE_FFMPEG=0

%CONDA_RUN% python %SCRIPT% %*

endlocal
