#!/usr/bin/env python3
"""
SubGenie GUI - Advanced Subtitle Generation & Translation Tool
===========================================================

Main GUI application for SubGenie subtitle generation and translation.
Provides user-friendly interface for all core functionality.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import json
import os
from pathlib import Path
import sys
import logging

# GUI components
from gui.main_interface import MainInterface
from gui.settings_manager import SettingsManager


# Core functionality
from core.gui_processor import GUIProcessor

# Configure appearance for modern look
ctk.set_appearance_mode("system")  # "system", "dark", "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class SubGenieApp:
    """Main application class for SubGenie GUI."""
    
    def __init__(self):
        """Initialize the main application."""
        self.app = ctk.CTk()
        self.app.title("SubGenie - Subtitle Generator")
        self.app.geometry("950x780")
        self.app.resizable(True, True)
        
        # Set minimum size
        self.app.minsize(900, 650)
        
        # Initialize components
        self.settings_manager = SettingsManager()
        self.processor = GUIProcessor(self.settings_manager)
        self.progress_dialog = None
        
        # Application state
        self.current_files = []
        self.processing = False
        
        # Setup UI
        self.setup_ui()
        self.load_settings()
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for GUI application."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "subgenie_gui.log"),
                logging.StreamHandler()
            ]
        )
    
    def setup_ui(self):
        """Setup the main user interface."""
        try:
            self.main_interface = MainInterface(self.app, self)
            logging.info("GUI interface initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize GUI: {e}")
            messagebox.showerror("Error", f"Failed to initialize GUI: {e}")
            sys.exit(1)
    
    def load_settings(self):
        """Load application settings."""
        try:
            self.settings_manager.load_settings()
            logging.info("Settings loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save application settings."""
        try:
            self.settings_manager.save_settings()
            logging.info("Settings saved successfully")
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
    
    def on_closing(self):
        """Handle application closing."""
        if self.processing:
            if messagebox.askyesno("Confirm Exit", "Processing is in progress. Are you sure you want to exit?"):
                self.processor.stop_processing()
            else:
                return
        
        self.save_settings()
        self.app.destroy()
    
    def run(self):
        """Run the main application."""
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.app.mainloop()

def main():
    """Main entry point for the application."""
    try:
        app = SubGenieApp()
        app.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Critical Error", f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()