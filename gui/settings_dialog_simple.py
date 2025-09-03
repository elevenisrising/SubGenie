"""
Simple Settings Dialog for SubGenie GUI
=======================================

Lightweight settings dialog to avoid GUI freezing.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from pathlib import Path

class SimpleSettingsDialog(ctk.CTkToplevel):
    """A simple settings dialog for theme selection."""
    
    def __init__(self, parent, settings_manager, language_manager):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.lang = language_manager
        
        self.title(self.lang.get_string("settings_title"))
        self.geometry("300x200")
        self.transient(parent)
        self.grab_set()
        
        # Get current theme and convert to display value
        current_theme = self.settings_manager.get_setting('theme', 'system')
        self.theme_var = tk.StringVar(value=self.get_theme_display_value(current_theme))
        
        self.create_widgets()
    
    def get_theme_display_value(self, system_value):
        """Convert system theme value to display value."""
        theme_mapping = {
            "light": self.lang.get_string("theme_light"),
            "dark": self.lang.get_string("theme_dark"), 
            "system": self.lang.get_string("theme_system")
        }
        return theme_mapping.get(system_value, self.lang.get_string("theme_system"))

    def create_widgets(self):
        """Create widgets for the settings dialog."""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Theme selection
        theme_frame = ctk.CTkFrame(main_frame)
        theme_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(theme_frame, text=self.lang.get_string("theme_label")).pack(side="left", padx=10)
        
        theme_menu = ctk.CTkOptionMenu(
            theme_frame, 
            variable=self.theme_var,
            values=[self.lang.get_string("theme_light"), self.lang.get_string("theme_dark"), self.lang.get_string("theme_system")]
        )
        theme_menu.pack(side="left", padx=10, expand=True, fill="x")
        
        # Button frame
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        save_btn = ctk.CTkButton(btn_frame, text=self.lang.get_string("save_btn"), command=self.save_settings)
        save_btn.pack(side="left", expand=True, padx=5)
        
        cancel_btn = ctk.CTkButton(btn_frame, text=self.lang.get_string("cancel_btn"), command=self.destroy)
        cancel_btn.pack(side="left", expand=True, padx=5)
        
    def save_settings(self):
        """Save settings and close the dialog."""
        selected_theme_display = self.theme_var.get()
        
        # Map display name back to system name
        theme_mapping = {
            self.lang.get_string("theme_light"): "light",
            self.lang.get_string("theme_dark"): "dark",
            self.lang.get_string("theme_system"): "system"
        }
        selected_theme = theme_mapping.get(selected_theme_display, "system")

        self.settings_manager.set_setting('theme', selected_theme)
        self.settings_manager.save_settings()
        ctk.set_appearance_mode(selected_theme)
        self.destroy()