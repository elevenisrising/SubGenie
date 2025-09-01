@echo off
REM =================================
REM  SubGenie GUI Launcher (Conda)
REM =================================

REM Set the name of your Conda environment
set CONDA_ENV_NAME=asr-env

echo Starting SubGenie GUI using Conda environment: %CONDA_ENV_NAME%...
echo This may take a moment...
echo.

REM Use 'conda run' to execute the python script directly in the specified environment.
REM This is the recommended and more reliable way to run a single task in Conda.
conda run -n %CONDA_ENV_NAME% python main_gui.py

echo.
echo Application has been closed. Press any key to exit.
pause >nul
