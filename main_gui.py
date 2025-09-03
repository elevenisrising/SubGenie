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
from gui.language_manager import LanguageManager


# Core functionality
from core.gui_processor import GUIProcessor

# Configure appearance for modern look
ctk.set_appearance_mode("system")  # "system", "dark", "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class SubGenieGUI(ctk.CTk):
    """Main application class for SubGenie."""

    def __init__(self):
        super().__init__()
        self.app = self # Add self-reference for consistency
        self.settings_manager = SettingsManager('settings.json')
        self.language_manager = LanguageManager() # Initialize LanguageManager
        self.setup_window()
        self.processor = GUIProcessor(self.settings_manager)
        self.current_files = []
        self.processing = False # Add missing attribute
        self.main_interface = MainInterface(self, self.language_manager) # Update constructor call
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_window(self):
        """Setup the main application window."""
        self.title(self.language_manager.get_string("app_title"))
        self.geometry("900x750")
        ctk.set_appearance_mode(self.settings_manager.get_setting('theme', 'system'))

    def update_title(self):
        """Update window title based on current language."""
        self.title(self.language_manager.get_string("app_title"))

    def update_status(self, message):
        """Update status bar message."""
        self.main_interface.status_var.set(message)
        self.update_idletasks()

    def on_closing(self):
        """Handle application closing."""
        if self.processing:
            if messagebox.askyesno(self.language_manager.get_string("confirm_exit_title"), self.language_manager.get_string("confirm_exit_message")):
                self.processor.stop_processing()
            else:
                return
        
        self.settings_manager.save_settings()
        self.destroy()
    
    def run(self):
        """Run the main application."""
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.app.mainloop()

def main():
    """Main entry point for the application."""
    try:
        app = SubGenieGUI()
        app.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Critical Error", f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()