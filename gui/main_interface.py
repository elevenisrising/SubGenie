"""
Main GUI Interface for SubGenie
===============================

Primary user interface components and layout.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from pathlib import Path
import threading
from typing import List, Optional

class MainInterface:
    """Main interface for SubGenie application."""
    
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.setup_layout()
        self.create_widgets()
        self.bind_events()
        self.ensure_directories()
    
    def setup_layout(self):
        """Setup main layout structure."""
        # Create main container
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)  # Content area more compressed
        self.main_frame.grid_rowconfigure(3, weight=4)  # Log area gets much more space
    
    def create_widgets(self):
        """Create all GUI widgets."""
        self.create_file_section()
        self.create_content_area()
        self.create_control_panel()
        self.create_status_bar()
    
    
    def create_file_section(self):
        """Create compact file selection section."""
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        file_frame.grid_columnconfigure(0, weight=1)
        
        # Top row: Files section with inline directory settings
        top_frame = ctk.CTkFrame(file_frame)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top_frame.grid_columnconfigure(2, weight=1)
        
        # File controls (left side)
        ctk.CTkLabel(top_frame, text="📁 Files:", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=5, sticky="w")
        
        btn_frame = ctk.CTkFrame(top_frame)
        btn_frame.grid(row=0, column=1, padx=5)
        
        ctk.CTkButton(btn_frame, text="Add Files", width=80, height=25,
                     command=self.add_files).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Add Folder", width=80, height=25,
                     command=self.add_folder).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Remove", width=60, height=25,
                     command=self.remove_files).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Clear", width=50, height=25,
                     command=self.clear_files).pack(side="left", padx=2)
        
        # Directory settings (right side)
        dirs_frame = ctk.CTkFrame(top_frame)
        dirs_frame.grid(row=0, column=2, padx=5, sticky="ew")
        dirs_frame.grid_columnconfigure(1, weight=1)
        dirs_frame.grid_columnconfigure(3, weight=1)
        
        # Source directory (first row)
        ctk.CTkLabel(dirs_frame, text="📁 Input:", 
                    font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, padx=2, sticky="w")
        
        self.source_dir_var = tk.StringVar(value=str(Path("input_audio").absolute()))
        source_entry = ctk.CTkEntry(dirs_frame, textvariable=self.source_dir_var, font=ctk.CTkFont(size=9))
        source_entry.grid(row=0, column=1, padx=2, sticky="ew")
        
        ctk.CTkLabel(dirs_frame, text="📂 Output:", 
                    font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=2, padx=2, sticky="w")
        
        self.output_dir_var = tk.StringVar(value=str(Path("output_subtitles").absolute()))
        output_entry = ctk.CTkEntry(dirs_frame, textvariable=self.output_dir_var, font=ctk.CTkFont(size=9))
        output_entry.grid(row=0, column=3, padx=2, sticky="ew")
        
        # Browse buttons (second row)
        ctk.CTkButton(dirs_frame, text="Browse", width=50, height=20,
                     command=self.browse_source_dir, font=ctk.CTkFont(size=9)).grid(row=1, column=1, padx=2, pady=2)
        ctk.CTkButton(dirs_frame, text="Open", width=50, height=20,
                     command=self.open_output_folder, font=ctk.CTkFont(size=9)).grid(row=1, column=3, padx=2, pady=2)
        
        # Compact file list
        list_frame = ctk.CTkFrame(file_frame)
        list_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
        self.file_listbox = tk.Listbox(list_frame, height=2, font=("Consolas", 9))
        self.file_listbox.pack(fill="x", padx=5, pady=3)
    
    def create_content_area(self):
        """Create main content area with tabs."""
        # Create tabview with more height - now has more space
        self.tabview = ctk.CTkTabview(self.main_frame, height=300)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Basic Processing Tab
        self.basic_tab = self.tabview.add("Basic Processing")
        self.create_basic_tab()
        
        # Advanced Tab
        self.advanced_tab = self.tabview.add("Advanced Options")
        self.create_advanced_tab()
        
        # Merge Subtitles Tab
        self.merge_tab = self.tabview.add("Merge Subtitles")
        self.create_merge_tab()
    
    def create_basic_tab(self):
        """Create basic processing options tab."""
        # Create scrollable frame for basic tab
        basic_scroll = ctk.CTkScrollableFrame(self.basic_tab)
        basic_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Language & Model settings (combined in one section)
        lang_model_frame = ctk.CTkFrame(basic_scroll)
        lang_model_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(lang_model_frame, text="Language & Model Settings", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        # First row: Source Language + Target Language  
        lang_row1 = ctk.CTkFrame(lang_model_frame)
        lang_row1.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(lang_row1, text="Source:").pack(side="left", padx=5)
        self.source_lang_var = tk.StringVar(value="Auto-detect")
        self.source_lang_combo = ctk.CTkComboBox(
            lang_row1,
            values=[
                "Auto-detect", "English", "Spanish", "French", "German", 
                "Japanese", "Korean", "Chinese (Simplified)", "Chinese (Traditional)"
            ],
            variable=self.source_lang_var,
            width=180
        )
        self.source_lang_combo.pack(side="left", padx=5)
        
        ctk.CTkLabel(lang_row1, text="Target:").pack(side="left", padx=(20, 5))
        self.target_lang_var = tk.StringVar(value="No Translation")
        self.target_lang_combo = ctk.CTkComboBox(
            lang_row1,
            values=[
                "No Translation", "Chinese (Simplified)", "Chinese (Traditional)", 
                "English", "Spanish", "French", "German", "Japanese", "Korean"
            ],
            variable=self.target_lang_var,
            width=180
        )
        self.target_lang_combo.pack(side="left", padx=5)
        
        # Second row: Model Size + Max Characters
        lang_row2 = ctk.CTkFrame(lang_model_frame)
        lang_row2.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(lang_row2, text="Whisper Model:").pack(side="left", padx=5)
        self.model_var = tk.StringVar(value="medium")
        self.model_combo = ctk.CTkComboBox(
            lang_row2,
            values=["tiny", "base", "small", "medium", "large"],
            variable=self.model_var,
            state="readonly",
            width=120
        )
        self.model_combo.pack(side="left", padx=5)
        
        ctk.CTkLabel(lang_row2, text="Max Chars:").pack(side="left", padx=(20, 5))
        self.max_chars_var = tk.StringVar(value="80")
        ctk.CTkEntry(lang_row2, textvariable=self.max_chars_var, width=60).pack(side="left", padx=5)
        
        # Output format
        output_frame = ctk.CTkFrame(basic_scroll)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(output_frame, text="Output Format", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        self.output_format_var = tk.StringVar(value="source")
        format_frame = ctk.CTkFrame(output_frame)
        format_frame.pack(fill="x", padx=10, pady=5)
        
        # Output format in one row
        ctk.CTkRadioButton(format_frame, text="Source Only", 
                          variable=self.output_format_var, value="source").pack(side="left", padx=10)
        ctk.CTkRadioButton(format_frame, text="Target Only", 
                          variable=self.output_format_var, value="target").pack(side="left", padx=10)
        ctk.CTkRadioButton(format_frame, text="Bilingual", 
                          variable=self.output_format_var, value="bilingual").pack(side="left", padx=10)
    
    def create_advanced_tab(self):
        """Create advanced options tab."""
        # Create scrollable frame for advanced tab
        advanced_scroll = ctk.CTkScrollableFrame(self.advanced_tab)
        advanced_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Translation settings
        trans_frame = ctk.CTkFrame(advanced_scroll)
        trans_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(trans_frame, text="🔄 Additional LLM Translation (Independent)", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        # Add explanation
        info_label = ctk.CTkLabel(
            trans_frame,
            text="💡 LLM translation will create separate output files, regardless of basic settings above",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        info_label.pack(anchor="w", padx=10, pady=2)
        
        self.translation_mode_var = tk.StringVar(value="free")
        self.translation_mode_var.trace("w", self.on_translation_mode_change)
        
        # Translation mode selection in one row
        mode_row = ctk.CTkFrame(trans_frame)
        mode_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkRadioButton(
            mode_row, 
            text="🆓 None", 
            variable=self.translation_mode_var, 
            value="free"
        ).pack(side="left", padx=10)
        
        ctk.CTkRadioButton(
            mode_row, 
            text="🏠 Local LLM", 
            variable=self.translation_mode_var, 
            value="local"
        ).pack(side="left", padx=10)
        
        ctk.CTkRadioButton(
            mode_row, 
            text="🌐 API LLM", 
            variable=self.translation_mode_var, 
            value="api"
        ).pack(side="left", padx=10)
        
        # Dynamic settings container
        self.dynamic_settings_frame = ctk.CTkFrame(trans_frame)
        self.dynamic_settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Initialize variables that are used in get_current_settings
        self.local_model_var = tk.StringVar(value="qwen2.5:7b")
        self.chunk_size_var = tk.StringVar(value="3")
        self.api_key_var = tk.StringVar(value="sk-u9VhdzHTG1dzEnWQTUzozkYfXhrkRpVisASUZdARQ0tORyQq")
        self.api_base_url_var = tk.StringVar(value="https://www.chataiapi.com/v1/chat/completions")
        self.api_model_var = tk.StringVar(value="gemini-2.5-pro")
        
        # Initialize with empty frame
        self.current_settings_widgets = []
        self.create_mode_specific_settings("free")
        
        # Set initial translation status
        self.set_translation_status("free")
        
        # LLM Prompt Settings
        prompt_frame = ctk.CTkFrame(advanced_scroll)
        prompt_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(prompt_frame, text="LLM Translation Prompts", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        # Prompt presets
        preset_frame = ctk.CTkFrame(prompt_frame)
        preset_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(preset_frame, text="Preset:").pack(side="left", padx=5)
        self.prompt_preset_var = tk.StringVar(value="Default")
        preset_combo = ctk.CTkComboBox(
            preset_frame,
            values=["Default", "Formal", "Casual", "Technical", "Literary", "Custom"],
            variable=self.prompt_preset_var,
            width=150,
            command=self.on_prompt_preset_change
        )
        preset_combo.pack(side="left", padx=5)
        
        # Custom prompt text area
        prompt_text_frame = ctk.CTkFrame(prompt_frame)
        prompt_text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(prompt_text_frame, text="Custom Prompt:").pack(anchor="w", padx=5, pady=2)
        
        self.custom_prompt_text = ctk.CTkTextbox(
            prompt_text_frame,
            height=80,
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="word"
        )
        self.custom_prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Set default prompt
        default_prompt = "You are a professional translator. Please translate the following subtitle text accurately while maintaining natural flow and cultural context. Keep the timing and formatting intact:"
        self.custom_prompt_text.insert("1.0", default_prompt)
    
    
    
    def create_mode_specific_settings(self, mode):
        """Create mode-specific settings UI."""
        # Clear existing widgets
        for widget in self.current_settings_widgets:
            widget.destroy()
        self.current_settings_widgets.clear()
        
        if mode == "free":
            # No additional settings needed for free mode
            info_label = ctk.CTkLabel(
                self.dynamic_settings_frame,
                text="💡 Only basic processing will run (no additional LLM translation)",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            info_label.pack(padx=10, pady=5)
            self.current_settings_widgets.append(info_label)
            
        elif mode == "local":
            # Local LLM settings
            local_frame = ctk.CTkFrame(self.dynamic_settings_frame)
            local_frame.pack(fill="x", padx=5, pady=5)
            self.current_settings_widgets.append(local_frame)
            
            # Model selection
            ctk.CTkLabel(local_frame, text="🏠 Local Model:", 
                        font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
            
            model_combo = ctk.CTkComboBox(
                local_frame,
                values=["qwen2.5:7b", "qwen2.5:14b", "llama3:8b", "codellama", "mistral"],
                variable=self.local_model_var,
                width=150
            )
            model_combo.pack(side="left", padx=5)
            
            # Chunk size
            ctk.CTkLabel(local_frame, text="Chunk Size:").pack(side="left", padx=(15, 5))
            chunk_entry = ctk.CTkEntry(local_frame, textvariable=self.chunk_size_var, width=50)
            chunk_entry.pack(side="left", padx=5)
            
        elif mode == "api":
            # API settings
            api_frame = ctk.CTkFrame(self.dynamic_settings_frame)
            api_frame.pack(fill="x", padx=5, pady=5)
            self.current_settings_widgets.append(api_frame)
            
            # API Configuration row 1
            api_row1 = ctk.CTkFrame(api_frame)
            api_row1.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(api_row1, text="🔑 API Key:", 
                        font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
            api_key_entry = ctk.CTkEntry(api_row1, textvariable=self.api_key_var, show="*", width=300)
            api_key_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            # API Configuration row 2
            api_row2 = ctk.CTkFrame(api_frame)
            api_row2.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(api_row2, text="🌐 Base URL:").pack(side="left", padx=5)
            base_url_entry = ctk.CTkEntry(api_row2, textvariable=self.api_base_url_var, width=300)
            base_url_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            # API Configuration row 3
            api_row3 = ctk.CTkFrame(api_frame)
            api_row3.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(api_row3, text="🤖 Model:").pack(side="left", padx=5)
            api_model_combo = ctk.CTkComboBox(
                api_row3,
                values=["gemini-2.5-pro", "gemini-1.5-pro", "deepseek-chat", "deepseek-coder", "claude-3.5-sonnet", "gpt-4o"],
                variable=self.api_model_var,
                width=150
            )
            api_model_combo.pack(side="left", padx=5)
            
            ctk.CTkLabel(api_row3, text="Chunk Size:").pack(side="left", padx=(15, 5))
            chunk_entry = ctk.CTkEntry(api_row3, textvariable=self.chunk_size_var, width=50)
            chunk_entry.pack(side="left", padx=5)
    
    def create_merge_tab(self):
        """Create merge subtitles tab."""
        merge_scroll = ctk.CTkScrollableFrame(self.merge_tab)
        merge_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Title
        ctk.CTkLabel(merge_scroll, text="🔗 Merge Subtitle Chunks", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        # Directory browser
        dir_frame = ctk.CTkFrame(merge_scroll)
        dir_frame.pack(fill="x", padx=20, pady=10)
        dir_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(dir_frame, text="Select Directory:", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.merge_dir_var = tk.StringVar()
        self.merge_dir_entry = ctk.CTkEntry(dir_frame, textvariable=self.merge_dir_var, 
                                          placeholder_text="Choose a directory to merge...")
        self.merge_dir_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        ctk.CTkButton(dir_frame, text="📁 Browse", width=80,
                     command=self.browse_merge_directory).grid(row=0, column=2, padx=10, pady=10)
        
        # Merge button
        ctk.CTkButton(
            merge_scroll,
            text="🔗 Merge All Chunks in Directory",
            command=self.simple_merge_chunks,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#2fa572", "#106a43"),
            width=300,
            height=50
        ).pack(pady=30)
        
        # Status Label
        self.merge_status_var = tk.StringVar(value="Select a directory containing chunk_*.srt files")
        ctk.CTkLabel(merge_scroll, textvariable=self.merge_status_var,
                    font=ctk.CTkFont(size=12), text_color="gray").pack(pady=5)
    
    def get_available_projects(self):
        """Get list of available projects for merging."""
        if hasattr(self.app, 'processor'):
            return self.app.processor.get_available_projects()
        return ["No projects found"]
    
    def refresh_merge_projects(self):
        """Refresh the project list."""
        projects = self.get_available_projects()
        self.merge_project_combo.configure(values=projects)
        if projects and projects[0] != "No projects found":
            self.merge_project_var.set(projects[0])
    
    def browse_merge_directory(self):
        """Browse for directory to merge."""
        directory = filedialog.askdirectory(
            title="Select Directory Containing Subtitle Chunks",
            initialdir="output_subtitles"
        )
        if directory:
            self.merge_dir_var.set(directory)
            # Show how many chunks found
            import os
            import re
            chunk_files = [f for f in os.listdir(directory) if re.match(r'chunk_(\d+)\.srt', f)]
            if chunk_files:
                self.merge_status_var.set(f"Found {len(chunk_files)} chunk files ready to merge")
            else:
                self.merge_status_var.set("No chunk_*.srt files found in selected directory")
    
    def simple_merge_chunks(self):
        """Merge all chunks in the selected directory using the GUI processor."""
        directory = self.merge_dir_var.get()
        
        if not directory:
            messagebox.showwarning("No Directory", "Please select a directory first.")
            return
            
        try:
            # Standardize path and find project root
            selected_path = Path(directory).resolve()
            output_subtitles_path = Path("output_subtitles").resolve()

            if not selected_path.is_relative_to(output_subtitles_path):
                messagebox.showerror("Invalid Directory", f"Please select a directory inside '{output_subtitles_path}'.")
                return

            relative_path = selected_path.relative_to(output_subtitles_path)
            
            if not relative_path.parts:
                messagebox.showerror("Invalid Directory", "Please select a project directory, not the root output folder.")
                return

            project_name = relative_path.parts[0]
            subfolder = relative_path.parts[1] if len(relative_path.parts) > 1 else "original"
            
            self.add_log(f"Starting merge for project '{project_name}', subfolder '{subfolder}'...")
            
            def merge_thread():
                try:
                    def log_callback(message):
                        self.parent.after(0, self.add_log, message)

                    success = self.app.processor.merge_chunks(project_name, subfolder, log_callback)
                    
                    if success:
                        self.parent.after(0, self.add_log, f"Successfully merged chunks for {project_name}/{subfolder}.")
                        self.parent.after(0, messagebox.showinfo, "Success", "Merge operation completed successfully.")
                    else:
                        self.parent.after(0, self.add_log, f"Failed to merge chunks for {project_name}/{subfolder}.")
                        self.parent.after(0, messagebox.showwarning, "Merge Failed", "The merge operation failed. Check logs for details.")
                except Exception as e:
                    self.parent.after(0, self.add_log, f"Error during merge thread: {e}")
            
            threading.Thread(target=merge_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.add_log(f"Error setting up merge: {e}")

    def create_control_panel(self):
        """Create main control buttons."""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Compact control buttons
        self.start_btn = ctk.CTkButton(
            control_frame,
            text="🚀 Start Processing",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=160,
            height=32,
            command=self.start_processing
        )
        self.start_btn.pack(side="left", padx=10, pady=5)
        
        # Stop button
        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="⏹️ Stop",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=80,
            height=32,
            state="disabled",
            command=self.stop_processing
        )
        self.stop_btn.pack(side="left", padx=5, pady=5)
        
        # Settings button
        ctk.CTkButton(
            control_frame,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=11),
            width=80,
            height=32,
            command=self.open_settings
        ).pack(side="right", padx=5, pady=5)
        
        # Open output folder button
        ctk.CTkButton(
            control_frame,
            text="📂 Output",
            font=ctk.CTkFont(size=11),
            width=80,
            height=32,
            command=self.open_output_folder
        ).pack(side="right", padx=5, pady=5)
    
    def create_status_bar(self):
        """Create status bar with log panel."""
        # Create a frame for the logs
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        self.main_frame.grid_rowconfigure(3, weight=4) # Make log area expandable

        # Log text area
        self.log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word"
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Status bar
        status_bar = ctk.CTkFrame(self.main_frame, height=25)
        status_bar.grid(row=4, column=0, sticky="ew", padx=5, pady=2)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(status_bar, textvariable=self.status_var, font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=10)
    
    def bind_events(self):
        """Bind GUI events."""
        # Bind keyboard shortcuts
        self.parent.bind("<Control-o>", lambda e: self.open_output_folder())
        self.parent.bind("<Control-O>", lambda e: self.open_output_folder())
        self.parent.bind("<F5>", lambda e: self.refresh_projects())
        self.parent.bind("<Control-r>", lambda e: self.refresh_projects())
        self.parent.bind("<Control-R>", lambda e: self.refresh_projects())
        
        # Focus to enable shortcuts
        self.parent.focus_set()
    
    def ensure_directories(self):
        """Ensure default directories exist."""
        try:
            Path("input_audio").mkdir(parents=True, exist_ok=True)
            Path("output_subtitles").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.add_log(f"Warning: Could not create default directories: {e}")
    
    def on_prompt_preset_change(self, value):
        """Handle prompt preset change."""
        prompts = {
            "Default": "You are a professional translator. Please translate the following subtitle text accurately while maintaining natural flow and cultural context. Keep the timing and formatting intact:",
            
            "Formal": "You are a professional translator specializing in formal documents. Please translate the following subtitle text with formal language and proper etiquette. Maintain accuracy and respect:",
            
            "Casual": "You are a friendly translator. Please translate the following subtitle text in a natural, casual way that sounds conversational and easy to understand:",
            
            "Technical": "You are a technical translator. Please translate the following subtitle text with precision, keeping technical terms accurate and maintaining professional terminology:",
            
            "Literary": "You are a literary translator. Please translate the following subtitle text with attention to style, tone, and artistic expression while maintaining the original meaning:",
            
            "Custom": ""
        }
        
        if value != "Custom":
            self.custom_prompt_text.delete("1.0", "end")
            self.custom_prompt_text.insert("1.0", prompts.get(value, prompts["Default"]))
    
    # Event handlers - basic implementations
    def add_files(self):
        """Add files to processing list."""
        filetypes = [
            ("All supported", "*.mp4 *.avi *.mov *.mp3 *.wav *.m4a"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("Audio files", "*.mp3 *.wav *.m4a *.aac"),
            ("All files", "*.*")
        ]
        
        # Use source directory as initial directory
        initial_dir = self.source_dir_var.get() if Path(self.source_dir_var.get()).exists() else ""
        
        files = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        if files:
            for file_path in files:
                if file_path not in self.app.current_files:
                    self.app.current_files.append(file_path)
                    self.file_listbox.insert(tk.END, Path(file_path).name)
    
    def add_folder(self):
        """Add folder to processing list."""
        # Use source directory as initial directory
        initial_dir = self.source_dir_var.get() if Path(self.source_dir_var.get()).exists() else ""
        
        folder_path = filedialog.askdirectory(
            title="Select Media Folder",
            initialdir=initial_dir
        )
        if folder_path:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.m4a', '.aac']
            folder = Path(folder_path)
            
            for ext in extensions:
                for file_path in folder.glob(f"*{ext}"):
                    file_str = str(file_path)
                    if file_str not in self.app.current_files:
                        self.app.current_files.append(file_str)
                        self.file_listbox.insert(tk.END, file_path.name)
    
    def remove_files(self):
        """Remove selected files from list."""
        selection = self.file_listbox.curselection()
        if selection:
            for index in reversed(selection):
                self.file_listbox.delete(index)
                if index < len(self.app.current_files):
                    self.app.current_files.pop(index)
    
    def clear_files(self):
        """Clear all files from list."""
        self.file_listbox.delete(0, tk.END)
        self.app.current_files.clear()
    
    def browse_source_dir(self):
        """Browse for source directory."""
        current_dir = self.source_dir_var.get() if Path(self.source_dir_var.get()).exists() else ""
        folder_path = filedialog.askdirectory(
            title="Select Source Directory",
            initialdir=current_dir
        )
        if folder_path:
            self.source_dir_var.set(folder_path)
            # Create directory if it doesn't exist
            Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    def open_output_folder(self):
        """Open output folder in file explorer."""
        import subprocess
        import platform
        
        output_path = Path(self.output_dir_var.get())
        output_path.mkdir(parents=True, exist_ok=True)
        
        if platform.system() == "Windows":
            subprocess.run(["explorer", str(output_path)])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(output_path)])
        else:  # Linux
            subprocess.run(["xdg-open", str(output_path)])
    
    def start_processing(self):
        """Start processing files."""
        if not self.app.current_files:
            messagebox.showwarning("No Files", "Please add files to process first.")
            return
        
        # Disable start button, enable stop button
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # Update status
        self.status_var.set("Processing...")
        self.clear_logs()
        self.add_log("Starting processing...")
        
        # Start processing in separate thread
        self.app.processing = True
        thread = threading.Thread(target=self._process_files_thread)
        thread.daemon = True
        thread.start()
    
    def stop_processing(self):
        """Stop processing files."""
        self.app.processing = False
        if hasattr(self.app, 'processor'):
            self.app.processor.stop_processing()
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Processing stopped")
        self.add_log("Processing stopped by user")
    
    def _process_files_thread(self):
        """Process files in separate thread."""
        try:
            # Get settings
            settings = self.get_current_settings()
            
            # Define log callback
            def log_callback(message):
                self.add_log(message)
            
            # Process files using processor
            if hasattr(self.app, 'processor'):
                self.app.processor.process_files(
                    self.app.current_files, 
                    settings, 
                    log_callback
                )
            self.status_var.set("Processing finished!")
        except Exception as e:
            self.add_log(f"Error: {str(e)}")
            self.status_var.set("Processing failed!")
        finally:
            # Reset buttons
            def _reset_buttons():
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
            
            self.parent.after(0, _reset_buttons)
            self.app.processing = False
    
    def get_current_settings(self):
        """Get current GUI settings."""
        target_lang = self.target_lang_var.get()
        translation_mode = self.translation_mode_var.get()
        enable_translation = target_lang != "No Translation"
        
        print(f"DEBUG - Target Language: '{target_lang}'")
        print(f"DEBUG - Translation Mode: '{translation_mode}'")
        print(f"DEBUG - Enable Translation: {enable_translation}")
        
        settings = {
            'model': self.model_var.get(),
            'source_language': self._map_language_to_code(self.source_lang_var.get()),
            'target_language': self._map_language_to_code(target_lang),
            'output_format': self.output_format_var.get(),
            'translation_mode': translation_mode,
            'local_model': self.local_model_var.get(),
            'max_chars': int(self.max_chars_var.get()) if self.max_chars_var.get().isdigit() else 80,
            'chunk_size': int(self.chunk_size_var.get()) if self.chunk_size_var.get().isdigit() else 3,
            'enable_translation': enable_translation,
            'custom_prompt': self.custom_prompt_text.get("1.0", "end-1c"),
            'prompt_preset': self.prompt_preset_var.get(),
            'source_directory': self.source_dir_var.get(),
            'output_directory': self.output_dir_var.get()
        }
        
        # Always print API settings debug info
        print(f"DEBUG - Has api_key_var: {hasattr(self, 'api_key_var')}")
        print(f"DEBUG - Has api_base_url_var: {hasattr(self, 'api_base_url_var')}")
        print(f"DEBUG - Has api_model_var: {hasattr(self, 'api_model_var')}")
        
        # Add API settings from GUI when using API translation mode
        if translation_mode == 'api':
            api_key = self.api_key_var.get()
            base_url = self.api_base_url_var.get()
            api_model = self.api_model_var.get()
            
            print(f"DEBUG - API Key: {'***' if api_key else 'EMPTY'}")
            print(f"DEBUG - Base URL: {base_url if base_url else 'EMPTY'}")
            print(f"DEBUG - API Model: {api_model}")
            
            settings.update({
                'api_key': api_key,
                'base_url': base_url,
                'api_model': api_model
            })
        
        return settings
    
    def _map_language_to_code(self, language_name):
        """Map language display name to code."""
        mapping = {
            "Auto-detect": "auto",
            "No Translation": "free",
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese (Simplified)": "zh-CN",
            "Chinese (Traditional)": "zh-TW"
        }
        return mapping.get(language_name, "auto")
    
    
    
    def add_log(self, message):
        """Add message to log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def _add():
            self.log_text.insert("end", log_message)
            self.log_text.see("end")
        
        self.parent.after(0, _add)
    
    def clear_logs(self):
        """Clear log panel."""
        self.log_text.delete("1.0", "end")
    
    
    def open_settings(self):
        """Open settings dialog."""
        try:
            from gui.settings_dialog_simple import SimpleSettingsDialog
            dialog = SimpleSettingsDialog(self.parent, self.app.settings_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open settings: {e}")
    
    def on_translation_mode_change(self, *args):
        """Update UI when translation mode changes."""
        mode = self.translation_mode_var.get()
        self.create_mode_specific_settings(mode)
        self.set_translation_status(mode)

    def set_translation_status(self, mode):
        """Update related UI elements based on translation mode."""
        if mode == "free":
            self.target_lang_var.set("No Translation")
            self.target_lang_combo.configure(state="disabled")
            self.output_format_var.set("source")
            for child in self.output_format_var.trace_info():
                if child[0] == 'w':
                    self.output_format_var.trace_vdelete('w', child[1])
        else:
            self.target_lang_combo.configure(state="readonly")
            if self.target_lang_var.get() == "No Translation":
                self.target_lang_var.set("Chinese (Simplified)")
            
            # Re-enable output format options
            # This logic might need more refinement if other modes have restrictions
            self.output_format_var.set("bilingual")