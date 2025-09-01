"""
Simple Settings Dialog for SubGenie GUI
=======================================

Lightweight settings dialog to avoid GUI freezing.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from pathlib import Path

class SimpleSettingsDialog:
    """Simple settings dialog without complex themes."""
    
    def __init__(self, parent, settings_manager):
        self.parent = parent
        self.settings_manager = settings_manager
        
        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x500")
        self.dialog.resizable(True, True)
        
        # Center on parent
        self.center_on_parent()
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self.setup_ui()
        
        # Load current settings
        self.load_current_settings()
    
    def center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        """Setup dialog UI."""
        # Main container
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="⚙️ SubGenie Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Scrollable frame for settings
        self.scroll_frame = ctk.CTkScrollableFrame(main_frame)
        self.scroll_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # API Settings Section
        self.create_api_section()
        
        # Paths Section
        self.create_paths_section()
        
        # Advanced Section
        self.create_advanced_section()
        
        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x")
        
        # Buttons
        ctk.CTkButton(
            button_frame,
            text="💾 Save & Close",
            command=self.save_and_close,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#2fa572", "#106a43"),
            hover_color=("#25a462", "#0d5936")
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="🔄 Reset",
            command=self.reset_settings,
            font=ctk.CTkFont(size=14),
            fg_color=("#d93d2b", "#b82e1f"),
            hover_color=("#c73426", "#a1261b")
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="❌ Cancel",
            command=self.close_dialog,
            font=ctk.CTkFont(size=14),
            fg_color="gray",
            hover_color="darkgray"
        ).pack(side="right", padx=5, pady=10)
    
    def create_api_section(self):
        """Create API settings section."""
        api_frame = ctk.CTkFrame(self.scroll_frame)
        api_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            api_frame, 
            text="🌐 API Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        # Relay API settings
        relay_frame = ctk.CTkFrame(api_frame)
        relay_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(relay_frame, text="Relay API Configuration", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)
        
        # Base URL
        url_container = ctk.CTkFrame(relay_frame)
        url_container.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(url_container, text="Base URL:", width=100).pack(side="left", padx=5)
        self.relay_url_var = tk.StringVar(value="https://www.chataiapi.com/v1/chat/completions")
        ctk.CTkEntry(url_container, textvariable=self.relay_url_var, width=400).pack(side="left", padx=5, expand=True, fill="x")
        
        # API Key
        key_container = ctk.CTkFrame(relay_frame)
        key_container.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(key_container, text="API Key:", width=100).pack(side="left", padx=5)
        self.relay_key_var = tk.StringVar()
        key_entry = ctk.CTkEntry(key_container, textvariable=self.relay_key_var, show="*", width=400)
        key_entry.pack(side="left", padx=5, expand=True, fill="x")
        
        # Default model
        model_container = ctk.CTkFrame(relay_frame)
        model_container.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(model_container, text="Default Model:", width=100).pack(side="left", padx=5)
        self.relay_model_var = tk.StringVar(value="gemini-2.5-pro")
        ctk.CTkOptionMenu(
            model_container,
            variable=self.relay_model_var,
            values=["gemini-2.5-pro", "gemini-1.5-pro", "deepseek-chat", "deepseek-coder"],
            width=200
        ).pack(side="left", padx=5)
    
    def create_paths_section(self):
        """Create paths section."""
        paths_frame = ctk.CTkFrame(self.scroll_frame)
        paths_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            paths_frame,
            text="📁 Directory Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        # Path settings
        self.path_vars = {}
        paths = [
            ("input_directory", "Input Directory:", "input_audio"),
            ("output_directory", "Output Directory:", "output_subtitles"),
            ("cache_directory", "Cache Directory:", ".cache"),
            ("logs_directory", "Logs Directory:", "logs")
        ]
        
        for key, label, default in paths:
            path_container = ctk.CTkFrame(paths_frame)
            path_container.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(path_container, text=label, width=150).pack(side="left", padx=5)
            
            self.path_vars[key] = tk.StringVar(value=default)
            path_entry = ctk.CTkEntry(path_container, textvariable=self.path_vars[key])
            path_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            ctk.CTkButton(
                path_container,
                text="📂 Browse",
                width=80,
                command=lambda k=key: self.browse_directory(k)
            ).pack(side="right", padx=5)
    
    def create_advanced_section(self):
        """Create advanced settings section."""
        advanced_frame = ctk.CTkFrame(self.scroll_frame)
        advanced_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            advanced_frame,
            text="⚙️ Advanced Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        # Checkboxes
        options_frame = ctk.CTkFrame(advanced_frame)
        options_frame.pack(fill="x", padx=10, pady=10)
        
        # Enable logging
        self.enable_logging_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_frame,
            text="Enable detailed logging",
            variable=self.enable_logging_var,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        # Auto-save settings
        self.auto_save_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_frame,
            text="Auto-save settings on changes",
            variable=self.auto_save_var,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        # Log level
        log_level_container = ctk.CTkFrame(options_frame)
        log_level_container.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(log_level_container, text="Log Level:", width=100).pack(side="left", padx=5)
        self.log_level_var = tk.StringVar(value="INFO")
        ctk.CTkOptionMenu(
            log_level_container,
            variable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=120
        ).pack(side="left", padx=5)
        
        # Default prompt settings
        prompt_container = ctk.CTkFrame(options_frame)
        prompt_container.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(prompt_container, text="Default Prompt Preset:", width=150).pack(side="left", padx=5)
        self.default_prompt_var = tk.StringVar(value="Default")
        ctk.CTkOptionMenu(
            prompt_container,
            variable=self.default_prompt_var,
            values=["Default", "Formal", "Casual", "Technical", "Literary"],
            width=120
        ).pack(side="left", padx=5)
    
    def browse_directory(self, key):
        """Browse for directory."""
        directory = filedialog.askdirectory(
            title=f"Select {key.replace('_', ' ').title()}",
            initialdir=self.path_vars[key].get()
        )
        if directory:
            self.path_vars[key].set(directory)
    
    def load_current_settings(self):
        """Load current settings into the dialog."""
        settings = self.settings_manager.current_settings
        
        # API settings
        relay_api = settings.get('api_settings', {}).get('relay_api', {})
        self.relay_url_var.set(relay_api.get('base_url', 'https://www.chataiapi.com/v1/chat/completions'))
        self.relay_key_var.set(relay_api.get('api_key', ''))
        self.relay_model_var.set(relay_api.get('default_model', 'gemini-2.5-pro'))
        
        # Paths
        paths = settings.get('paths', {})
        for key, var in self.path_vars.items():
            var.set(paths.get(key, var.get()))
        
        # Advanced
        advanced = settings.get('advanced', {})
        self.enable_logging_var.set(advanced.get('enable_logging', True))
        self.auto_save_var.set(advanced.get('auto_save_settings', True))
        self.log_level_var.set(advanced.get('log_level', 'INFO'))
        self.default_prompt_var.set(advanced.get('default_prompt_preset', 'Default'))
    
    def save_and_close(self):
        """Save settings and close dialog."""
        try:
            # API settings
            self.settings_manager.set_setting('api_settings.relay_api.base_url', self.relay_url_var.get())
            self.settings_manager.set_setting('api_settings.relay_api.api_key', self.relay_key_var.get())
            self.settings_manager.set_setting('api_settings.relay_api.default_model', self.relay_model_var.get())
            
            # Paths
            for key, var in self.path_vars.items():
                self.settings_manager.set_setting(f'paths.{key}', var.get())
            
            # Advanced
            self.settings_manager.set_setting('advanced.enable_logging', self.enable_logging_var.get())
            self.settings_manager.set_setting('advanced.auto_save_settings', self.auto_save_var.get())
            self.settings_manager.set_setting('advanced.log_level', self.log_level_var.get())
            self.settings_manager.set_setting('advanced.default_prompt_preset', self.default_prompt_var.get())
            
            # Save to file
            self.settings_manager.save_settings()
            
            messagebox.showinfo("Success", "Settings saved successfully!", parent=self.dialog)
            self.close_dialog()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}", parent=self.dialog)
    
    def reset_settings(self):
        """Reset settings to defaults."""
        if messagebox.askyesno("Confirm Reset", "Reset all settings to defaults?", parent=self.dialog):
            self.settings_manager.reset_settings()
            self.load_current_settings()
            messagebox.showinfo("Reset Complete", "Settings reset to defaults", parent=self.dialog)
    
    def close_dialog(self):
        """Close the dialog."""
        self.dialog.grab_release()
        self.dialog.destroy()