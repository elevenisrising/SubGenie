#!/usr/bin/env python3
"""
Build Script for SubGenie GUI Executable
========================================

Creates a standalone executable for SubGenie using PyInstaller.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    if description:
        print(f"Description: {description}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_dependencies():
    """Check if required tools are installed."""
    print("Checking dependencies...")
    
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        if not run_command([sys.executable, "-m", "pip", "install", "pyinstaller"]):
            return False
    
    # Check customtkinter
    try:
        import customtkinter
        print(f"✅ CustomTkinter {customtkinter.__version__} found")
    except ImportError:
        print("❌ CustomTkinter not found. Installing GUI requirements...")
        if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_gui.txt"]):
            return False
    
    return True

def create_build_directories():
    """Create necessary build directories."""
    print("Creating build directories...")
    
    build_dir = Path("build")
    dist_dir = Path("dist")
    
    # Clean previous builds
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("🗑️ Cleaned build directory")
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print("🗑️ Cleaned dist directory")
    
    # Create fresh directories
    build_dir.mkdir(exist_ok=True)
    dist_dir.mkdir(exist_ok=True)
    
    return True

def build_executable(build_type="onefile"):
    """Build the executable using PyInstaller."""
    print(f"Building executable ({build_type})...")
    
    # Base PyInstaller command
    cmd = [
        "pyinstaller",
        "--name", "SubGenie",
        "--windowed",  # Hide console window
        "--icon", "assets/icon.ico" if Path("assets/icon.ico").exists() else None,
        
        # Include data files
        "--add-data", "gui;gui",
        "--add-data", "core;core",
        "--add-data", "main.py;.",
        "--add-data", "llm_translate.py;.",
        "--add-data", "relay_api_translate.py;.",
        "--add-data", "reprocess_chunk.py;.",
        "--add-data", "requirements_gui.txt;.",
        
        # Include FFmpeg if available
        "--add-binary", "ffmpeg.exe;." if Path("ffmpeg.exe").exists() else None,
        
        # Hidden imports for GUI
        "--hidden-import", "customtkinter",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL._tkinter_finder",
        
        # Core hidden imports
        "--hidden-import", "whisper",
        "--hidden-import", "spacy",
        "--hidden-import", "nltk",
        "--hidden-import", "srt",
        
        # Exclude unnecessary modules
        "--exclude-module", "matplotlib",
        "--exclude-module", "scipy",
        "--exclude-module", "pandas",
        
        "main_gui.py"
    ]
    
    # Add onefile option if requested
    if build_type == "onefile":
        cmd.insert(1, "--onefile")
    
    # Remove None values
    cmd = [arg for arg in cmd if arg is not None]
    
    return run_command(cmd, "Building SubGenie executable")

def create_installer_script():
    """Create a simple installer script."""
    installer_content = '''@echo off
echo SubGenie Installation
echo ====================
echo.

REM Create directories
if not exist "input_audio" mkdir input_audio
if not exist "output_subtitles" mkdir output_subtitles
if not exist "logs" mkdir logs

echo Created necessary directories.
echo.

REM Check for FFmpeg
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: FFmpeg not found in PATH
    echo Please install FFmpeg for full functionality
    echo Download from: https://ffmpeg.org/download.html
    echo.
)

echo Installation completed!
echo.
echo Usage:
echo 1. Place audio/video files in the 'input_audio' folder
echo 2. Run SubGenie.exe
echo 3. Select files and configure settings
echo 4. Click "Start Processing"
echo.
echo Results will be saved in the 'output_subtitles' folder
echo.
pause
'''
    
    with open("dist/install.bat", "w") as f:
        f.write(installer_content)
    
    print("✅ Created installer script")

def create_readme():
    """Create README for distribution."""
    readme_content = '''# SubGenie - Standalone Distribution

## Quick Start
1. Run `install.bat` to set up directories
2. Double-click `SubGenie.exe` to launch the application
3. Add your audio/video files using the "Add Files" button
4. Configure translation settings
5. Click "Start Processing"

## Features
- 🎯 Drag-and-drop file support
- 🌍 Multiple translation modes (Free, Local LLM, Commercial APIs)
- 🤖 Advanced AI-powered translation
- 📝 Multiple output formats (SRT, VTT, ASS, TXT, JSON)
- 🔧 Chunk reprocessing and management
- ⚙️ Customizable settings

## Requirements
- Windows 10/11 (64-bit)
- Internet connection for translation services
- FFmpeg (recommended, for video processing)

## File Structure
```
SubGenie/
├── SubGenie.exe          # Main application
├── install.bat           # Setup script
├── input_audio/          # Place your audio/video files here
├── output_subtitles/     # Processed subtitles appear here
└── logs/                # Application logs
```

## Troubleshooting
- If FFmpeg errors occur, install FFmpeg and add to PATH
- For API translation, configure your API keys in Settings
- Check logs/ folder for detailed error information

## Support
- GitHub: https://github.com/elevenisrising/SubGenie
- Issues: Report bugs and feature requests on GitHub

Built with ❤️ using Python, Whisper, and spaCy
'''
    
    with open("dist/README.txt", "w") as f:
        f.write(readme_content)
    
    print("✅ Created distribution README")

def main():
    """Main build process."""
    parser = argparse.ArgumentParser(description="Build SubGenie executable")
    parser.add_argument("--type", choices=["onefile", "onedir"], default="onefile",
                       help="Build type (default: onefile)")
    parser.add_argument("--no-upx", action="store_true",
                       help="Disable UPX compression")
    
    args = parser.parse_args()
    
    print("🚀 SubGenie Build Process")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return 1
    
    # Step 2: Create build directories
    if not create_build_directories():
        print("❌ Failed to create build directories")
        return 1
    
    # Step 3: Build executable
    if not build_executable(args.type):
        print("❌ Executable build failed")
        return 1
    
    # Step 4: Create additional files
    create_installer_script()
    create_readme()
    
    print("\n✅ Build completed successfully!")
    print(f"📦 Executable: dist/SubGenie.exe")
    print(f"📁 Distribution folder: dist/")
    
    # Show file sizes
    exe_path = Path("dist/SubGenie.exe")
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"📏 Executable size: {size_mb:.1f} MB")
    
    print("\n🎉 Ready for distribution!")
    return 0

if __name__ == "__main__":
    sys.exit(main())