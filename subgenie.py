#!/usr/bin/env python3
"""
SubGenie - Main Entry Point
============================

Command-line entry point for SubGenie subtitle generation tool.
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SubGenie - Subtitle Generation Tool")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("file", nargs="?", help="Media file to process")
    parser.add_argument("--model", default="medium", help="Whisper model to use")
    parser.add_argument("--target-language", default="zh-CN", help="Target language for translation")
    
    args = parser.parse_args()
    
    if args.gui or not args.file:
        # Launch GUI
        import main_gui
        main_gui.main()
    else:
        # Run command line processing
        sys.path.insert(0, str(Path(__file__).parent / "src" / "processing"))
        import main
        
        # Convert args to sys.argv format for main.py
        sys.argv = [
            "main.py",
            args.file,
            "--model", args.model,
            "--target_language", args.target_language
        ]
        main.main()

if __name__ == "__main__":
    main()