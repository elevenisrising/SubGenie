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
import os
import re
from gui.settings_dialog_simple import SimpleSettingsDialog


class MainInterface:
    """Main interface for SubGenie application."""
    
    def __init__(self, app, language_manager):
        self.app = app
        self.lang = language_manager
        # Store references to all components that need language updates
        self.translatable_components = {}
        self.setup_layout()
        self.create_widgets()
        self.bind_events()
        self.ensure_directories()
        # Set initial output format constraints
        self.on_target_language_change()
    
    def store_translatable_component(self, component, text_key):
        """Store a reference to a component that needs text translation."""
        if text_key not in self.translatable_components:
            self.translatable_components[text_key] = []
        self.translatable_components[text_key].append(component)
    
    def setup_layout(self):
        """Setup main layout structure."""
        # Create main container
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)  # Content area more compressed
        self.main_frame.grid_rowconfigure(3, weight=4)  # Log area gets much more space

        # Header frame
        self.header_frame = ctk.CTkFrame(self.main_frame)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=0)
    
    def create_widgets(self):
        """Create all GUI widgets."""
        self.create_language_switcher()
        self.create_file_section()
        self.create_content_area()
        self.create_control_panel()
        self.create_status_bar()
    
    def create_file_section(self):
        """Create file selection section with left controls and right scrollable file list."""
        import os
        
        # Main file section with more height
        file_frame = ctk.CTkFrame(self.header_frame, height=180)
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        file_frame.grid_columnconfigure(1, weight=1)
        file_frame.grid_propagate(False)
        
        # Left side - Controls and settings (fixed width)
        controls_frame = ctk.CTkFrame(file_frame, width=280)
        controls_frame.grid(row=0, column=0, sticky="ns", padx=(5, 2), pady=5)
        controls_frame.grid_propagate(False)
        
        # Title only
        files_label = ctk.CTkLabel(controls_frame, text=self.lang.get_string("files_section_title"), 
                                   font=ctk.CTkFont(size=14, weight="bold"))
        files_label.pack(anchor="w", padx=5, pady=(5, 10))
        self.store_translatable_component(files_label, "files_section_title")
        
        # File operation buttons - 2x2 grid layout
        btn_frame = ctk.CTkFrame(controls_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        
        add_files_btn = ctk.CTkButton(btn_frame, text=self.lang.get_string("add_files"), 
                                      height=20, width=120, command=self.add_files)
        add_files_btn.grid(row=0, column=0, padx=2, pady=1, sticky="ew")
        self.store_translatable_component(add_files_btn, "add_files")
        
        add_folder_btn = ctk.CTkButton(btn_frame, text=self.lang.get_string("add_folder"), 
                                       height=20, width=120, command=self.add_folder)
        add_folder_btn.grid(row=0, column=1, padx=2, pady=1, sticky="ew")
        self.store_translatable_component(add_folder_btn, "add_folder")
        
        remove_btn = ctk.CTkButton(btn_frame, text=self.lang.get_string("remove"), 
                                   height=20, width=120, command=self.remove_files)
        remove_btn.grid(row=1, column=0, padx=2, pady=1, sticky="ew")
        self.store_translatable_component(remove_btn, "remove")
        
        clear_btn = ctk.CTkButton(btn_frame, text=self.lang.get_string("clear"), 
                                  height=20, width=120, command=self.clear_files)
        clear_btn.grid(row=1, column=1, padx=2, pady=1, sticky="ew")
        self.store_translatable_component(clear_btn, "clear")
        
        # Directory settings - below buttons
        dir_frame = ctk.CTkFrame(controls_frame)
        dir_frame.pack(fill="x", padx=5, pady=(10, 5))
        dir_frame.grid_columnconfigure(1, weight=1)
        
        # Use centralized path management
        try:
            from src.utils.constants import get_input_dir, get_output_dir
            default_input = str(get_input_dir())
            default_output = str(get_output_dir())
        except ImportError:
            # Fallback to working directory relative paths
            working_dir = Path.cwd()
            default_input = str(working_dir / "input_audio")
            default_output = str(working_dir / "output_subtitles")
        
        self.source_dir_var = tk.StringVar(value=default_input)
        self.output_dir_var = tk.StringVar(value=default_output)
        
        # Input directory
        input_label = ctk.CTkLabel(dir_frame, text=self.lang.get_string("input_dir"), font=ctk.CTkFont(size=10))
        input_label.grid(row=0, column=0, sticky="w", padx=3, pady=2)
        self.store_translatable_component(input_label, "input_dir")
        
        input_entry = ctk.CTkEntry(dir_frame, textvariable=self.source_dir_var, font=ctk.CTkFont(size=9), height=24)
        input_entry.grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        
        browse_input_btn = ctk.CTkButton(dir_frame, text=self.lang.get_string("browse"), 
                                         width=50, height=24, command=self.browse_source_dir,
                                         font=ctk.CTkFont(size=9))
        browse_input_btn.grid(row=0, column=2, padx=3, pady=2)
        self.store_translatable_component(browse_input_btn, "browse")
        
        # Output directory
        output_label = ctk.CTkLabel(dir_frame, text=self.lang.get_string("output_dir"), font=ctk.CTkFont(size=10))
        output_label.grid(row=1, column=0, sticky="w", padx=3, pady=2)
        self.store_translatable_component(output_label, "output_dir")
        
        output_entry = ctk.CTkEntry(dir_frame, textvariable=self.output_dir_var, font=ctk.CTkFont(size=9), height=24)
        output_entry.grid(row=1, column=1, sticky="ew", padx=3, pady=2)
        
        browse_output_btn = ctk.CTkButton(dir_frame, text=self.lang.get_string("browse"), 
                                          width=50, height=24, command=self.browse_output_dir,
                                          font=ctk.CTkFont(size=9))
        browse_output_btn.grid(row=1, column=2, padx=3, pady=2)
        self.store_translatable_component(browse_output_btn, "browse")
        
        # Right side - Scrollable file list (expandable)
        list_frame = ctk.CTkFrame(file_frame)
        list_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # File list title with language button
        title_header = ctk.CTkFrame(list_frame)
        title_header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        title_header.grid_columnconfigure(0, weight=1)
        
        list_title = ctk.CTkLabel(title_header, text=self.lang.get_string("selected_files_title"), font=ctk.CTkFont(size=12, weight="bold"))
        list_title.grid(row=0, column=0, sticky="w")
        self.store_translatable_component(list_title, "selected_files_title")
        
        # Language toggle button - compact square design
        self.lang_button = ctk.CTkButton(
            title_header,
            text=self._get_language_button_text(),
            width=32,
            height=28,
            command=self.toggle_language,
            fg_color=("#1E1E1E", "#2D2D30"),  # VS Code dark theme inspired
            hover_color=("#007ACC", "#0E639C"),  # Blue accent on hover
            text_color=("#CCCCCC", "#FFFFFF"),
            border_width=0,
            corner_radius=6,
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.lang_button.grid(row=0, column=1, sticky="e", padx=(10, 0))
        
        # Scrollable listbox container
        listbox_container = ctk.CTkFrame(list_frame)
        listbox_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Create listbox with scrollbar - ensure proper scrolling
        self.file_listbox = tk.Listbox(
            listbox_container, 
            font=("Consolas", 9),
            selectmode=tk.EXTENDED,  # Allow multiple selections
            activestyle="none"
        )
        scrollbar = tk.Scrollbar(listbox_container, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Pack listbox and scrollbar
        self.file_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_content_area(self):
        """Create main content area with tabs."""
        # Create tabview with more height - now has more space
        self.tabview = ctk.CTkTabview(self.main_frame, height=300)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Basic Processing Tab
        self.basic_tab = self.tabview.add(self.lang.get_string("tab_basic"))
        self.create_basic_tab(self.basic_tab)

        # Advanced Tab
        self.advanced_tab = self.tabview.add(self.lang.get_string("tab_advanced"))
        self.create_advanced_tab(self.advanced_tab)

        # Merge Subtitles Tab
        self.merge_tab = self.tabview.add(self.lang.get_string("tab_merge"))
        self.create_merge_tab(self.merge_tab)

    def create_language_switcher(self):
        """Language switcher is now created in the file list title area."""  
        pass

    def _get_language_button_text(self):
        """Get compact text for language button"""
        if self.lang.current_language == "en":
            return "中"  # Just Chinese character
        else:
            return "EN"  # Just English abbreviation
    
    def toggle_language(self):
        """Toggle language between en and zh."""
        if self.lang.current_language == "en":
            self.lang.set_language("zh")
        else:
            self.lang.set_language("en")
        self.update_language_strings()


    def update_language_strings(self):
        """Update all UI text strings without recreating widgets."""
        try:
            # Update language button text with new styled text
            self.lang_button.configure(text=self._get_language_button_text())
            
            # Update window title
            if hasattr(self.app, 'update_title'):
                self.app.update_title()
            
            # Update all stored translatable components
            self.update_all_translatable_components()
            
            # Update all combo boxes with new language values
            self.update_combo_boxes()
            
            # Update status text
            self.update_status_bar()
            
            # Update other variable text elements
            self.update_variables()
            
            # Update dynamic settings components
            self.update_dynamic_settings()
            
            # Re-apply translation status to ensure target language dropdown state is correct
            if hasattr(self, 'translation_mode_var'):
                current_mode = self.translation_mode_var.get()
                self.set_translation_status(current_mode)
            else:
                self.set_translation_status("free")
            
            print(f"Language successfully switched to: {self.lang.current_language}")
            
        except Exception as e:
            print(f"Error updating language strings: {e}")
    
    def update_all_translatable_components(self):
        """Update all stored translatable components."""
        try:
            for text_key, components in self.translatable_components.items():
                new_text = self.lang.get_string(text_key)
                for component in components:
                    try:
                        component.configure(text=new_text)
                    except Exception as e:
                        print(f"Error updating component for key {text_key}: {e}")
        except Exception as e:
            print(f"Error in update_all_translatable_components: {e}")
    
    def update_combo_boxes(self):
        """Update combo box options with new language."""
        try:
            # Create language mappings for consistent translation
            source_lang_mapping = {
                # English keys to display values
                "auto_detect": self.lang.get_string("auto_detect"),
                "english": self.lang.get_string("english"), 
                "spanish": self.lang.get_string("spanish"),
                "french": self.lang.get_string("french"),
                "german": self.lang.get_string("german"), 
                "japanese": self.lang.get_string("japanese"),
                "korean": self.lang.get_string("korean"),
                "chinese_simplified": self.lang.get_string("chinese_simplified"),
                "chinese_traditional": self.lang.get_string("chinese_traditional")
            }
            
            target_lang_mapping = {
                "no_translation": self.lang.get_string("no_translation"),
                "chinese_simplified": self.lang.get_string("chinese_simplified"),
                "chinese_traditional": self.lang.get_string("chinese_traditional"),
                "english": self.lang.get_string("english"),
                "spanish": self.lang.get_string("spanish"), 
                "french": self.lang.get_string("french"),
                "german": self.lang.get_string("german"),
                "japanese": self.lang.get_string("japanese"),
                "korean": self.lang.get_string("korean")
            }
            
            # Update source language combo
            if hasattr(self, 'source_lang_combo'):
                current_value = self.source_lang_var.get()
                # Find which key this current value corresponds to
                current_key = self.find_language_key(current_value, source_lang_mapping)
                
                new_values = list(source_lang_mapping.values())
                self.source_lang_combo.configure(values=new_values)
                
                # Set the translated version of the current selection
                if current_key and current_key in source_lang_mapping:
                    self.source_lang_var.set(source_lang_mapping[current_key])
                else:
                    self.source_lang_var.set(new_values[0])  # Default to first
                    
            # Update target language combo  
            if hasattr(self, 'target_lang_combo'):
                current_value = self.target_lang_var.get()
                current_key = self.find_language_key(current_value, target_lang_mapping)
                
                new_values = list(target_lang_mapping.values())
                self.target_lang_combo.configure(values=new_values)
                
                # Set the translated version of the current selection
                if current_key and current_key in target_lang_mapping:
                    self.target_lang_var.set(target_lang_mapping[current_key])
                else:
                    self.target_lang_var.set(new_values[0])  # Default to first
                    
        except Exception as e:
            print(f"Error updating combo boxes: {e}")
    
    def find_language_key(self, display_value, mapping):
        """Find the language key for a given display value."""
        # Check current mapping first
        for key, value in mapping.items():
            if value == display_value:
                return key
        
        # Check reverse mapping (for previous language values)
        reverse_mappings = {
            # English -> keys
            "Auto-detect": "auto_detect",
            "English": "english", 
            "Spanish": "spanish",
            "French": "french", 
            "German": "german",
            "Japanese": "japanese",
            "Korean": "korean",
            "Chinese (Simplified)": "chinese_simplified",
            "Chinese (Traditional)": "chinese_traditional",
            "No Translation": "no_translation",
            # Chinese -> keys  
            "自动检测": "auto_detect",
            "英语": "english",
            "西班牙语": "spanish", 
            "法语": "french",
            "德语": "german",
            "日语": "japanese",
            "韩语": "korean", 
            "简体中文": "chinese_simplified",
            "繁体中文": "chinese_traditional",
            "不翻译": "no_translation"
        }
        
        return reverse_mappings.get(display_value)
    
    
    def update_status_bar(self):
        """Update status bar text."""
        try:
            if hasattr(self, 'status_var'):
                current_status = self.status_var.get()
                # Only update if it's the "Ready" status
                if "Ready" in current_status or "就绪" in current_status:
                    self.status_var.set(self.lang.get_string("status_ready"))
        except Exception as e:
            print(f"Error updating status bar: {e}")
    
    def update_variables(self):
        """Update other variable text elements."""
        try:
            if hasattr(self, 'merge_status_var'):
                current = self.merge_status_var.get()
                if "Select a directory" in current or "请选择" in current:
                    self.merge_status_var.set(self.lang.get_string("merge_status_label"))
        except Exception as e:
            print(f"Error updating variables: {e}")
    
    def update_dynamic_settings(self):
        """Update dynamic settings components."""
        try:
            # Re-create the current mode settings with new language
            if hasattr(self, 'translation_mode_var'):
                current_mode = self.translation_mode_var.get()
                self.create_mode_specific_settings(current_mode)
        except Exception as e:
            print(f"Error updating dynamic settings: {e}")
    
    def create_basic_tab(self, tab):
        """Create basic processing options tab."""
        # Create scrollable frame for basic tab
        basic_scroll = ctk.CTkScrollableFrame(tab)
        basic_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Language & Model settings (combined in one section)
        lang_model_frame = ctk.CTkFrame(basic_scroll)
        lang_model_frame.pack(fill="x", padx=10, pady=10)
        
        lang_model_label = ctk.CTkLabel(lang_model_frame, text=self.lang.get_string("lang_model_settings_title"), 
                    font=ctk.CTkFont(size=16, weight="bold"))
        lang_model_label.pack(anchor="w", padx=10, pady=5)
        self.store_translatable_component(lang_model_label, "lang_model_settings_title")
        
        # First row: Source Language + Target Language  
        lang_row1 = ctk.CTkFrame(lang_model_frame)
        lang_row1.pack(fill="x", padx=10, pady=5)
        
        source_label = ctk.CTkLabel(lang_row1, text=self.lang.get_string("source_lang_label"))
        source_label.pack(side="left", padx=5)
        self.store_translatable_component(source_label, "source_lang_label")
        # Use centralized defaults for consistency
        try:
            from src.utils.constants import DEFAULT_SOURCE_LANGUAGE
            default_source = self.lang.get_string("auto_detect") if DEFAULT_SOURCE_LANGUAGE == "auto" else DEFAULT_SOURCE_LANGUAGE
        except ImportError:
            default_source = self.lang.get_string("auto_detect")
        self.source_lang_var = tk.StringVar(value=default_source)

        source_lang_names = [
            self.lang.get_string("auto_detect"), self.lang.get_string("english"), self.lang.get_string("spanish"), 
            self.lang.get_string("french"), self.lang.get_string("german"), self.lang.get_string("japanese"), 
            self.lang.get_string("korean"), self.lang.get_string("chinese_simplified"), self.lang.get_string("chinese_traditional")
        ]

        self.source_lang_combo = ctk.CTkComboBox(
            lang_row1,
            values=source_lang_names,
            variable=self.source_lang_var,
            width=180
        )
        self.source_lang_combo.pack(side="left", padx=5)
        
        target_label = ctk.CTkLabel(lang_row1, text=self.lang.get_string("target_lang_label"))
        target_label.pack(side="left", padx=(20, 5))
        self.store_translatable_component(target_label, "target_lang_label")
        # Use centralized defaults for target language
        try:
            from src.utils.constants import DEFAULT_TARGET_LANGUAGE
            # Map language code to GUI display string
            if DEFAULT_TARGET_LANGUAGE == "zh-CN":
                default_target = self.lang.get_string("chinese_simplified")
            elif DEFAULT_TARGET_LANGUAGE == "zh-TW":
                default_target = self.lang.get_string("chinese_traditional")
            elif DEFAULT_TARGET_LANGUAGE == "en":
                default_target = self.lang.get_string("english")
            else:
                default_target = self.lang.get_string("no_translation")
        except ImportError:
            default_target = self.lang.get_string("no_translation")
        self.target_lang_var = tk.StringVar(value=default_target)

        target_lang_names = [
            self.lang.get_string("no_translation"), self.lang.get_string("chinese_simplified"), self.lang.get_string("chinese_traditional"), 
            self.lang.get_string("english"), self.lang.get_string("spanish"), self.lang.get_string("french"), 
            self.lang.get_string("german"), self.lang.get_string("japanese"), self.lang.get_string("korean")
        ]

        self.target_lang_combo = ctk.CTkComboBox(
            lang_row1,
            values=target_lang_names,
            variable=self.target_lang_var,
            width=180,
            command=self.on_target_language_change
        )
        self.target_lang_combo.pack(side="left", padx=5)
        
        # Second row: Model Size + Max Characters
        lang_row2 = ctk.CTkFrame(lang_model_frame)
        lang_row2.pack(fill="x", padx=10, pady=5)
        
        whisper_label = ctk.CTkLabel(lang_row2, text=self.lang.get_string("whisper_model_label"))
        whisper_label.pack(side="left", padx=5)
        self.store_translatable_component(whisper_label, "whisper_model_label")
        self.model_var = tk.StringVar(value="medium")
        self.model_combo = ctk.CTkComboBox(
            lang_row2,
            values=["tiny", "base", "small", "medium", "large"],
            variable=self.model_var,
            state="readonly",
            width=120
        )
        self.model_combo.pack(side="left", padx=5)
        
        max_chars_label = ctk.CTkLabel(lang_row2, text=self.lang.get_string("max_chars_label"))
        max_chars_label.pack(side="left", padx=(20, 5))
        self.store_translatable_component(max_chars_label, "max_chars_label")
        self.max_chars_var = tk.StringVar(value="80")
        ctk.CTkEntry(lang_row2, textvariable=self.max_chars_var, width=60).pack(side="left", padx=5)
        
        chunk_time_label = ctk.CTkLabel(lang_row2, text=self.lang.get_string("chunk_time_label"))
        chunk_time_label.pack(side="left", padx=(20, 5))
        self.store_translatable_component(chunk_time_label, "chunk_time_label")
        self.chunk_time_var = tk.StringVar(value="30")
        ctk.CTkEntry(lang_row2, textvariable=self.chunk_time_var, width=60).pack(side="left", padx=5)
        
        # Third row: Segmentation Strategy
        lang_row3 = ctk.CTkFrame(lang_model_frame)
        lang_row3.pack(fill="x", padx=10, pady=5)
        
        seg_label = ctk.CTkLabel(lang_row3, text=self.lang.get_string("segmentation_strategy_label"))
        seg_label.pack(side="left", padx=(0, 5))
        self.store_translatable_component(seg_label, "segmentation_strategy_label")
        
        self.segmentation_strategy_var = tk.StringVar(value="spacy")
        segmentation_combo = ctk.CTkComboBox(
            lang_row3,
            values=[
                self.lang.get_string("spacy_segmentation"),
                self.lang.get_string("whisper_segmentation")
            ],
            variable=self.segmentation_strategy_var,
            width=150
        )
        segmentation_combo.pack(side="left", padx=5)

        # Output format
        output_frame = ctk.CTkFrame(basic_scroll)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        output_title_label = ctk.CTkLabel(output_frame, text=self.lang.get_string("output_format_title"), 
                    font=ctk.CTkFont(size=16, weight="bold"))
        output_title_label.pack(anchor="w", padx=10, pady=5)
        self.store_translatable_component(output_title_label, "output_format_title")
        
        self.output_format_var = tk.StringVar(value="source")
        format_frame = ctk.CTkFrame(output_frame)
        format_frame.pack(fill="x", padx=10, pady=5)
        
        # Output format in one row
        source_radio = ctk.CTkRadioButton(format_frame, text=self.lang.get_string("source_only_radio"), 
                          variable=self.output_format_var, value="source")
        source_radio.pack(side="left", padx=10)
        self.store_translatable_component(source_radio, "source_only_radio")
        
        target_radio = ctk.CTkRadioButton(format_frame, text=self.lang.get_string("target_only_radio"), 
                          variable=self.output_format_var, value="target")
        target_radio.pack(side="left", padx=10)
        self.store_translatable_component(target_radio, "target_only_radio")
        
        bilingual_radio = ctk.CTkRadioButton(format_frame, text=self.lang.get_string("bilingual_radio"), 
                          variable=self.output_format_var, value="bilingual")
        bilingual_radio.pack(side="left", padx=10)
        self.store_translatable_component(bilingual_radio, "bilingual_radio")
        
        # Store radio buttons for enabling/disabling
        self.source_radio = source_radio
        self.target_radio = target_radio
        self.bilingual_radio = bilingual_radio
        
        # Audio Settings
        audio_frame = ctk.CTkFrame(basic_scroll)
        audio_frame.pack(fill="x", padx=10, pady=10)
        
        audio_title_label = ctk.CTkLabel(audio_frame, text=self.lang.get_string("audio_preprocessing_title"), 
                    font=ctk.CTkFont(size=16, weight="bold"))
        audio_title_label.pack(anchor="w", padx=10, pady=5)
        self.store_translatable_component(audio_title_label, "audio_preprocessing_title")
        
        # Checkboxes
        check_row = ctk.CTkFrame(audio_frame)
        check_row.pack(fill="x", padx=10, pady=2)

        self.no_preprocessing_var = tk.BooleanVar(value=False)
        no_preprocess_check = ctk.CTkCheckBox(check_row, text=self.lang.get_string("disable_all_preprocessing_checkbox"), variable=self.no_preprocessing_var)
        no_preprocess_check.pack(side="left", padx=10)
        self.store_translatable_component(no_preprocess_check, "disable_all_preprocessing_checkbox")
        
        self.no_normalize_var = tk.BooleanVar(value=False)
        no_normalize_check = ctk.CTkCheckBox(check_row, text=self.lang.get_string("disable_normalization_checkbox"), variable=self.no_normalize_var)
        no_normalize_check.pack(side="left", padx=10)
        self.store_translatable_component(no_normalize_check, "disable_normalization_checkbox")
        
        self.no_denoise_var = tk.BooleanVar(value=False)
        no_denoise_check = ctk.CTkCheckBox(check_row, text=self.lang.get_string("disable_denoise_checkbox"), variable=self.no_denoise_var)
        no_denoise_check.pack(side="left", padx=10)
        self.store_translatable_component(no_denoise_check, "disable_denoise_checkbox")
        
        self.use_ai_vocal_separation_var = tk.BooleanVar(value=False)
        ai_vocal_check = ctk.CTkCheckBox(check_row, text=self.lang.get_string("use_ai_vocal_separation_checkbox"), variable=self.use_ai_vocal_separation_var)
        ai_vocal_check.pack(side="left", padx=10)
        self.store_translatable_component(ai_vocal_check, "use_ai_vocal_separation_checkbox")
        
        # Parameters
        param_row = ctk.CTkFrame(audio_frame)
        param_row.pack(fill="x", padx=10, pady=5)
        
        dbfs_label = ctk.CTkLabel(param_row, text=self.lang.get_string("target_dbfs_label"))
        dbfs_label.pack(side="left", padx=(10, 5))
        self.store_translatable_component(dbfs_label, "target_dbfs_label")
        self.target_dbfs_var = tk.StringVar(value="-20.0")
        ctk.CTkEntry(param_row, textvariable=self.target_dbfs_var, width=60).pack(side="left", padx=5)

        denoise_label = ctk.CTkLabel(param_row, text=self.lang.get_string("denoise_strength_label"))
        denoise_label.pack(side="left", padx=(20, 5))
        self.store_translatable_component(denoise_label, "denoise_strength_label")
        self.denoise_strength_var = tk.StringVar(value="0.5")
        ctk.CTkEntry(param_row, textvariable=self.denoise_strength_var, width=60).pack(side="left", padx=5)
        
        # Performance settings row
        perf_row = ctk.CTkFrame(audio_frame)
        perf_row.pack(fill="x", padx=10, pady=5)
        
        workers_label = ctk.CTkLabel(perf_row, text=self.lang.get_string("parallel_workers_label"))
        workers_label.pack(side="left", padx=(10, 5))
        self.store_translatable_component(workers_label, "parallel_workers_label")
        self.max_workers_var = tk.StringVar(value="1")
        ctk.CTkEntry(perf_row, textvariable=self.max_workers_var, width=60).pack(side="left", padx=5)
        
        self.no_parallel_var = tk.BooleanVar(value=False)
        no_parallel_check = ctk.CTkCheckBox(perf_row, text=self.lang.get_string("disable_parallel_checkbox"), variable=self.no_parallel_var)
        no_parallel_check.pack(side="left", padx=(20, 5))
        self.store_translatable_component(no_parallel_check, "disable_parallel_checkbox")
        
    def on_target_language_change(self, value=None):
        """Handle target language changes and enable/disable output format options."""
        try:
            target_lang = self.target_lang_var.get()
            no_translation_text = self.lang.get_string("no_translation")
            
            if target_lang == no_translation_text:
                # User selected "No Translation" - disable target and bilingual options
                self.target_radio.configure(state="disabled")
                self.bilingual_radio.configure(state="disabled")
                
                # If currently selected format is not available, switch to source
                current_format = self.output_format_var.get()
                if current_format in ["target", "bilingual"]:
                    self.output_format_var.set("source")
            else:
                # User selected a translation language - enable all options
                self.target_radio.configure(state="normal")
                self.bilingual_radio.configure(state="normal")
                
        except Exception as e:
            # Silently handle any errors to avoid disrupting the UI
            print(f"Error in target language change handler: {e}")

    def create_advanced_tab(self, tab):
        """Create advanced options tab."""
        # Create scrollable frame for advanced tab
        advanced_scroll = ctk.CTkScrollableFrame(tab)
        advanced_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Translation settings
        trans_frame = ctk.CTkFrame(advanced_scroll)
        trans_frame.pack(fill="x", padx=10, pady=10)
        
        llm_title_label = ctk.CTkLabel(trans_frame, text=self.lang.get_string("advanced_llm_translation_title"), 
                    font=ctk.CTkFont(size=16, weight="bold"))
        llm_title_label.pack(anchor="w", padx=10, pady=5)
        self.store_translatable_component(llm_title_label, "advanced_llm_translation_title")
        
        # Add explanation
        info_label = ctk.CTkLabel(
            trans_frame,
            text=self.lang.get_string("llm_translation_info"),
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        info_label.pack(anchor="w", padx=10, pady=2)
        self.store_translatable_component(info_label, "llm_translation_info")
        
        self.translation_mode_var = tk.StringVar(value="none")
        
        # Translation mode selection in one row
        mode_row = ctk.CTkFrame(trans_frame)
        mode_row.pack(fill="x", padx=10, pady=5)
        
        none_radio = ctk.CTkRadioButton(
            mode_row, 
            text=self.lang.get_string("none_radio"), 
            variable=self.translation_mode_var, 
            value="none"
        )
        none_radio.pack(side="left", padx=10)
        self.store_translatable_component(none_radio, "none_radio")
        
        local_radio = ctk.CTkRadioButton(
            mode_row, 
            text=self.lang.get_string("local_llm_radio"), 
            variable=self.translation_mode_var, 
            value="local"
        )
        local_radio.pack(side="left", padx=10)
        self.store_translatable_component(local_radio, "local_llm_radio")
        
        api_radio = ctk.CTkRadioButton(
            mode_row, 
            text=self.lang.get_string("api_llm_radio"), 
            variable=self.translation_mode_var, 
            value="api"
        )
        api_radio.pack(side="left", padx=10)
        self.store_translatable_component(api_radio, "api_llm_radio")
        
        # Dynamic settings container
        self.dynamic_settings_frame = ctk.CTkFrame(trans_frame)
        self.dynamic_settings_frame.pack(fill="x", padx=10, pady=5)
        
        # This list MUST be initialized before on_translation_mode_change is called.
        self.current_settings_widgets = []
        
        # Initialize variables using centralized constants
        try:
            from src.utils.constants import (
                DEFAULT_LOCAL_MODEL, GUI_DEFAULT_CHUNK_SIZE, DEFAULT_API_MODEL
            )
            default_local_model = DEFAULT_LOCAL_MODEL
            default_chunk_size = str(GUI_DEFAULT_CHUNK_SIZE)
            default_api_model = DEFAULT_API_MODEL
        except ImportError:
            default_local_model = "qwen2.5:7b"
            default_chunk_size = "5"
            default_api_model = "gemini-2.5-pro"
            
        self.local_model_var = tk.StringVar(value=default_local_model)
        self.chunk_size_var = tk.StringVar(value=default_chunk_size)
        self.api_key_var = tk.StringVar(value="")
        self.api_base_url_var = tk.StringVar(value="")
        self.api_model_var = tk.StringVar(value=default_api_model)
        
        # Manually call the handler to set the initial UI state. This is now safe.
        self.on_translation_mode_change()
        # Add the trace *after* initial setup. It will only fire on user-driven changes.
        self.translation_mode_var.trace("w", self.on_translation_mode_change)
        
        # LLM Prompt Settings
        prompt_frame = ctk.CTkFrame(advanced_scroll)
        prompt_frame.pack(fill="x", padx=10, pady=10)
        
        prompts_title_label = ctk.CTkLabel(prompt_frame, text=self.lang.get_string("llm_prompts_title"), 
                    font=ctk.CTkFont(size=16, weight="bold"))
        prompts_title_label.pack(anchor="w", padx=10, pady=5)
        self.store_translatable_component(prompts_title_label, "llm_prompts_title")
        
        # Prompt presets
        preset_frame = ctk.CTkFrame(prompt_frame)
        preset_frame.pack(fill="x", padx=10, pady=5)
        
        preset_label = ctk.CTkLabel(preset_frame, text=self.lang.get_string("preset_label"))
        preset_label.pack(side="left", padx=5)
        self.store_translatable_component(preset_label, "preset_label")
        self.prompt_preset_var = tk.StringVar(value=self.lang.get_string("prompt_preset_default"))
        preset_values = [
            self.lang.get_string("prompt_preset_default"),
            self.lang.get_string("prompt_preset_formal"),
            self.lang.get_string("prompt_preset_casual"),
            self.lang.get_string("prompt_preset_technical"),
            self.lang.get_string("prompt_preset_literary"),
            self.lang.get_string("prompt_preset_custom")
        ]
        preset_combo = ctk.CTkComboBox(
            preset_frame,
            values=preset_values,
            variable=self.prompt_preset_var,
            width=150,
            command=self.on_prompt_preset_change
        )
        preset_combo.pack(side="left", padx=5)
        
        # Custom prompt text area
        prompt_text_frame = ctk.CTkFrame(prompt_frame)
        prompt_text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        custom_prompt_label = ctk.CTkLabel(prompt_text_frame, text=self.lang.get_string("custom_prompt_label"))
        custom_prompt_label.pack(anchor="w", padx=5, pady=2)
        self.store_translatable_component(custom_prompt_label, "custom_prompt_label")
        
        self.custom_prompt_text = ctk.CTkTextbox(
            prompt_text_frame,
            height=80,
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="word"
        )
        self.custom_prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Set default prompt
        default_prompt = self.lang.get_string("default_prompt")
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
                text=self.lang.get_string("only_basic_processing_info"),
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
            ctk.CTkLabel(local_frame, text=self.lang.get_string("local_model_label"), 
                        font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
            
            model_combo = ctk.CTkComboBox(
                local_frame,
                values=["qwen2.5:7b", "qwen2.5:14b", "llama3:8b", "codellama", "mistral"],
                variable=self.local_model_var,
                width=150
            )
            model_combo.pack(side="left", padx=5)
            
            # Chunk size
            ctk.CTkLabel(local_frame, text=self.lang.get_string("chunk_size_label")).pack(side="left", padx=(15, 5))
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
            
            ctk.CTkLabel(api_row1, text=self.lang.get_string("api_key_label"), 
                        font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
            api_key_entry = ctk.CTkEntry(api_row1, textvariable=self.api_key_var, show="*", width=300)
            api_key_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            # API Configuration row 2
            api_row2 = ctk.CTkFrame(api_frame)
            api_row2.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(api_row2, text=self.lang.get_string("base_url_label")).pack(side="left", padx=5)
            base_url_entry = ctk.CTkEntry(api_row2, textvariable=self.api_base_url_var, width=300)
            base_url_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            # API Configuration row 3
            api_row3 = ctk.CTkFrame(api_frame)
            api_row3.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(api_row3, text=self.lang.get_string("api_model_label")).pack(side="left", padx=5)
            api_model_combo = ctk.CTkComboBox(
                api_row3,
                values=["gemini-2.5-pro", "gemini-1.5-pro", "deepseek-chat", "deepseek-coder", "claude-3.5-sonnet", "gpt-4o"],
                variable=self.api_model_var,
                width=150
            )
            api_model_combo.pack(side="left", padx=5)
            
            ctk.CTkLabel(api_row3, text=self.lang.get_string("chunk_size_label")).pack(side="left", padx=(15, 5))
            chunk_entry = ctk.CTkEntry(api_row3, textvariable=self.chunk_size_var, width=50)
            chunk_entry.pack(side="left", padx=5)
    
    def create_merge_tab(self, tab):
        """Create merge subtitles tab."""
        merge_scroll = ctk.CTkScrollableFrame(tab)
        merge_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Title
        merge_title_label = ctk.CTkLabel(merge_scroll, text=self.lang.get_string("merge_subtitle_chunks_title"), 
                    font=ctk.CTkFont(size=18, weight="bold"))
        merge_title_label.pack(pady=20)
        self.store_translatable_component(merge_title_label, "merge_subtitle_chunks_title")
        
        # Directory browser
        dir_frame = ctk.CTkFrame(merge_scroll)
        dir_frame.pack(fill="x", padx=20, pady=10)
        dir_frame.grid_columnconfigure(1, weight=1)
        
        select_dir_label = ctk.CTkLabel(dir_frame, text=self.lang.get_string("select_directory_label"), 
                    font=ctk.CTkFont(size=14, weight="bold"))
        select_dir_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.store_translatable_component(select_dir_label, "select_directory_label")
        
        self.merge_dir_var = tk.StringVar()
        self.merge_dir_entry = ctk.CTkEntry(dir_frame, textvariable=self.merge_dir_var, 
                                          placeholder_text=self.lang.get_string("choose_dir_placeholder"))
        self.merge_dir_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        browse_merge_btn = ctk.CTkButton(dir_frame, text=self.lang.get_string("browse_btn"), width=80,
                     command=self.browse_merge_directory)
        browse_merge_btn.grid(row=0, column=2, padx=10, pady=10)
        self.store_translatable_component(browse_merge_btn, "browse_btn")
        
        # Merge button
        merge_chunks_btn = ctk.CTkButton(
            merge_scroll,
            text=self.lang.get_string("merge_all_chunks_btn"),
            command=self.simple_merge_chunks,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("#2fa572", "#106a43"),
            width=300,
            height=50
        )
        merge_chunks_btn.pack(pady=30)
        self.store_translatable_component(merge_chunks_btn, "merge_all_chunks_btn")
        
        # Status Label
        self.merge_status_var = tk.StringVar(value=self.lang.get_string("merge_status_label"))
        ctk.CTkLabel(merge_scroll, textvariable=self.merge_status_var,
                    font=ctk.CTkFont(size=12), text_color="gray").pack(pady=5)
    
    def get_available_projects(self):
        """Get list of available projects for merging."""
        if hasattr(self.app, 'processor'):
            return self.app.processor.get_available_projects()
        return [self.lang.get_string("no_projects_found")]
    
    def refresh_merge_projects(self):
        """Refresh the project list."""
        projects = self.get_available_projects()
        self.merge_project_combo.configure(values=projects)
        if projects and projects[0] != self.lang.get_string("no_projects_found"):
            self.merge_project_var.set(projects[0])
    
    def browse_merge_directory(self):
        """Browse for directory to merge."""
        try:
            from src.utils.constants import get_output_dir
            initial_dir = str(get_output_dir())
        except ImportError:
            initial_dir = "output_subtitles"
            
        directory = filedialog.askdirectory(
            title=self.lang.get_string("select_directory_title"),
            initialdir=initial_dir
        )
        if directory:
            self.merge_dir_var.set(directory)
            # Show how many chunks found
            chunk_files = [f for f in os.listdir(directory) if re.match(r'chunk_(\d+)\.srt', f)]
            if chunk_files:
                self.merge_status_var.set(f"Found {len(chunk_files)} chunk files ready to merge")
            else:
                self.merge_status_var.set("No chunk_*.srt files found in selected directory")
    
    def simple_merge_chunks(self):
        """Merge all chunks in the selected directory using the GUI processor."""
        directory = self.merge_dir_var.get()
        
        if not directory:
            messagebox.showwarning(self.lang.get_string("no_directory_warning_title"), self.lang.get_string("no_directory_warning_message"))
            return
            
        try:
            # Standardize path and find project root
            selected_path = Path(directory).resolve()
            try:
                from src.utils.constants import get_output_dir
                output_subtitles_path = get_output_dir().resolve()
            except ImportError:
                output_subtitles_path = Path("output_subtitles").resolve()

            if not selected_path.is_relative_to(output_subtitles_path):
                messagebox.showerror(self.lang.get_string("invalid_directory_error_title"), self.lang.get_string("invalid_directory_error_message").format(output_subtitles_path=output_subtitles_path))
                return

            relative_path = selected_path.relative_to(output_subtitles_path)
            
            if not relative_path.parts:
                messagebox.showerror(self.lang.get_string("invalid_directory_error_title"), self.lang.get_string("invalid_project_directory_error_message"))
                return

            project_name = relative_path.parts[0]
            subfolder = relative_path.parts[1] if len(relative_path.parts) > 1 else None
            
            self.add_log(f"Starting merge for project '{project_name}', subfolder '{subfolder}'...")
            
            def merge_thread():
                try:
                    def log_callback(message):
                        self.app.after(0, self.add_log, message)

                    success = self.app.processor.merge_chunks(project_name, subfolder, log_callback)
                    
                    if success:
                        self.app.after(0, self.add_log, f"Successfully merged chunks for {project_name}/{subfolder}.")
                        self.app.after(0, messagebox.showinfo, self.lang.get_string("success_title"), self.lang.get_string("merge_success_message"))
                    else:
                        self.app.after(0, self.add_log, f"Failed to merge chunks for {project_name}/{subfolder}.")
                        self.app.after(0, messagebox.showwarning, self.lang.get_string("merge_failed_title"), self.lang.get_string("merge_failed_message"))
                except Exception as e:
                    self.app.after(0, self.add_log, f"Error during merge thread: {e}")
            
            threading.Thread(target=merge_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror(self.lang.get_string("error_title"), self.lang.get_string("unexpected_error_message").format(e=e))
            self.add_log(f"Error setting up merge: {e}")

    def create_control_panel(self):
        """Create main control buttons."""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Compact control buttons
        self.start_btn = ctk.CTkButton(
            control_frame,
            text=self.lang.get_string("start_processing_btn"),
            font=ctk.CTkFont(size=14, weight="bold"),
            width=160,
            height=32,
            command=self.start_processing
        )
        self.start_btn.pack(side="left", padx=10, pady=5)
        self.store_translatable_component(self.start_btn, "start_processing_btn")
        
        # Stop button
        self.stop_btn = ctk.CTkButton(
            control_frame,
            text=self.lang.get_string("stop_btn"),
            font=ctk.CTkFont(size=12, weight="bold"),
            width=80,
            height=32,
            state="disabled",
            command=self.stop_processing
        )
        self.stop_btn.pack(side="left", padx=5, pady=5)
        self.store_translatable_component(self.stop_btn, "stop_btn")
        
        
        # Open output folder button
        output_btn = ctk.CTkButton(
            control_frame,
            text=self.lang.get_string("output_btn"),
            font=ctk.CTkFont(size=11),
            width=80,
            height=32,
            command=self.open_output_folder
        )
        output_btn.pack(side="right", padx=5, pady=5)
        self.store_translatable_component(output_btn, "output_btn")
    
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
        
        self.status_var = tk.StringVar(value=self.lang.get_string("status_ready"))
        self.status_label = ctk.CTkLabel(status_bar, textvariable=self.status_var, font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=10)
    
    def bind_events(self):
        """Bind GUI events."""
        # Bind keyboard shortcuts
        self.app.bind("<Control-o>", lambda e: self.open_output_folder())
        self.app.bind("<Control-O>", lambda e: self.open_output_folder())
        self.app.bind("<F5>", lambda e: self.refresh_projects())
        self.app.bind("<Control-r>", lambda e: self.refresh_projects())
        self.app.bind("<Control-R>", lambda e: self.refresh_projects())
        
        # Focus to enable shortcuts
        self.app.focus_set()
    
    def ensure_directories(self):
        """Ensure default directories exist."""
        try:
            from src.utils.constants import get_input_dir, get_output_dir
            get_input_dir().mkdir(parents=True, exist_ok=True)
            get_output_dir().mkdir(parents=True, exist_ok=True)
        except ImportError:
            # Fallback to hardcoded paths
            Path("input_audio").mkdir(parents=True, exist_ok=True)
            Path("output_subtitles").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.add_log(f"Warning: Could not create default directories: {e}")
    
    def on_prompt_preset_change(self, value):
        """Handle prompt preset change."""
        prompts = {
            self.lang.get_string("prompt_preset_default"): self.lang.get_string("default_prompt"),
            self.lang.get_string("prompt_preset_formal"): self.lang.get_string("formal_prompt"),
            self.lang.get_string("prompt_preset_casual"): self.lang.get_string("casual_prompt"),
            self.lang.get_string("prompt_preset_technical"): self.lang.get_string("technical_prompt"),
            self.lang.get_string("prompt_preset_literary"): self.lang.get_string("literary_prompt"),
            self.lang.get_string("prompt_preset_custom"): ""
        }
        
        if value != self.lang.get_string("prompt_preset_custom"):
            self.custom_prompt_text.delete("1.0", "end")
            self.custom_prompt_text.insert("1.0", prompts.get(value, prompts[self.lang.get_string("prompt_preset_default")]))
    
    # Event handlers - basic implementations
    def add_files(self):
        """Add files to processing list."""
        filetypes = [
            (self.lang.get_string("all_supported_files"), "*.mp4 *.avi *.mov *.mp3 *.wav *.m4a"),
            (self.lang.get_string("video_files"), "*.mp4 *.avi *.mov *.mkv"),
            (self.lang.get_string("audio_files"), "*.mp3 *.wav *.m4a *.aac"),
            (self.lang.get_string("all_files"), "*.*")
        ]
        
        # Use source directory as initial directory
        initial_dir = self.source_dir_var.get() if Path(self.source_dir_var.get()).exists() else ""
        
        files = filedialog.askopenfilenames(
            title=self.lang.get_string("select_media_files_title"),
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
            title=self.lang.get_string("select_media_folder_title"),
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
            title=self.lang.get_string("select_source_directory_title"),
            initialdir=current_dir
        )
        if folder_path:
            self.source_dir_var.set(folder_path)
            # Create directory if it doesn't exist
            Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        current_dir = self.output_dir_var.get() if Path(self.output_dir_var.get()).exists() else ""
        folder_path = filedialog.askdirectory(
            title=self.lang.get_string("select_output_directory_title"),
            initialdir=current_dir
        )
        if folder_path:
            self.output_dir_var.set(folder_path)
            # Create directory if it doesn't exist
            Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    def _get_segmentation_strategy_value(self):
        """Get the internal strategy value from GUI selection."""
        if not hasattr(self, 'segmentation_strategy_var'):
            return "spacy"
        
        display_value = self.segmentation_strategy_var.get()
        
        # Map display values to internal values
        strategy_mapping = {
            self.lang.get_string("spacy_segmentation"): "spacy",
            self.lang.get_string("whisper_segmentation"): "whisper"
        }
        
        # Also handle English/Chinese reverse mapping
        reverse_mapping = {
            "spaCy Grammar": "spacy",
            "Whisper Segments": "whisper", 
            "spaCy语法": "spacy",
            "Whisper分句": "whisper"
        }
        
        # Try current language mapping first
        result = strategy_mapping.get(display_value)
        if result:
            return result
            
        # Try reverse mapping
        result = reverse_mapping.get(display_value)
        if result:
            return result
            
        # Default fallback
        return "spacy"
    
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
            messagebox.showwarning(self.lang.get_string("no_files_warning_title"), self.lang.get_string("no_files_warning_message"))
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
        if messagebox.askyesno(self.lang.get_string("confirm_stop_title"), self.lang.get_string("confirm_stop_message")):
            self.app.processor.stop_processing()
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set(self.lang.get_string("processing_stopped"))
        self.add_log(self.lang.get_string("processing_stopped_by_user"))
    
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
            
            self.app.after(0, _reset_buttons)
            self.app.processing = False
    
    def get_current_settings(self):
        """Get current processing settings from the GUI."""
        target_lang = self.target_lang_var.get()
        translation_mode = self.translation_mode_var.get()
        enable_translation = target_lang != self.lang.get_string("no_translation")
        
        print(f"DEBUG - Target Language: '{target_lang}'")
        print(f"DEBUG - Translation Mode: '{translation_mode}'")
        print(f"DEBUG - Enable Translation: {enable_translation}")
        
        source_lang_val = self.source_lang_var.get()
        target_lang_val = self.target_lang_var.get()
        
        # Handle "Auto-detect" and "No Translation" which might be translated
        if source_lang_val == self.lang.get_string("auto_detect"):
            source_lang_code = "auto"
        else:
            source_lang_code = self._map_language_to_code(source_lang_val)

        if target_lang_val == self.lang.get_string("no_translation"):
            target_lang_code = "none"  # Use 'none' instead of 'free'
        else:
            target_lang_code = self._map_language_to_code(target_lang_val)

        settings = {
            # Core processing settings
            "model": self.model_var.get(),
            "source_language": source_lang_code,
            "target_language": target_lang_code,
            "output_format": self.output_format_var.get(),
            
            # Translation settings
            "translation_mode": translation_mode,
            "local_model": self.local_model_var.get(),
            "chunk_size": int(self.chunk_size_var.get()) if self.chunk_size_var.get().isdigit() else 3,
            "enable_translation": enable_translation,
            "custom_prompt": self.custom_prompt_text.get("1.0", "end-1c"),
            "prompt_preset": self.prompt_preset_var.get(),
            
            # Subtitle formatting
            "max_chars": int(self.max_chars_var.get()) if self.max_chars_var.get().isdigit() else 80,
            "segmentation_strategy": self._get_segmentation_strategy_value(),
            
            # Audio chunking (fixed parameter names)
            "chunk_time": int(self.chunk_time_var.get()) if self.chunk_time_var.get().isdigit() else 30,
            "long_audio_threshold": 15,  # Could be made configurable
            
            # Directory settings
            "source_directory": self.source_dir_var.get(),
            "output_directory": self.output_dir_var.get(),
            
            # Audio preprocessing settings
            "no_audio_preprocessing": self.no_preprocessing_var.get(),
            "no_normalize_audio": self.no_normalize_var.get(),
            "no_denoise": self.no_denoise_var.get(),
            "use_ai_vocal_separation": self.use_ai_vocal_separation_var.get(),
            "target_dbfs": float(self.target_dbfs_var.get()) if self.target_dbfs_var.get() else -20.0,
            "noise_reduction_strength": float(self.denoise_strength_var.get()) if self.denoise_strength_var.get() else 0.5,
            
            # Performance settings
            "max_workers": int(self.max_workers_var.get()) if self.max_workers_var.get().isdigit() else 2,
            "no_parallel_processing": self.no_parallel_var.get(),
            
            # Translation control - only enable if user actually chose a target language
            "translate_during_detection": enable_translation,
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
        # This reversed lookup is more robust
        language_key = next((k for k, v in self.lang.current_strings.items() if v == language_name), None)
        
        mapping = {
            "auto_detect": "auto",
            "no_translation": "none",
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "japanese": "ja",
            "korean": "ko",
            "chinese_simplified": "zh-CN",
            "chinese_traditional": "zh-TW"
        }
        return mapping.get(language_key, "auto")
    
    
    
    def add_log(self, message):
        """Add message to log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def _add():
            self.log_text.insert("end", log_message)
            self.log_text.see("end")
        
        self.app.after(0, _add)
    
    def clear_logs(self):
        """Clear log panel."""
        self.log_text.delete("1.0", "end")
    
    
    def open_settings(self):
        """Open settings dialog."""
        try:
            dialog = SimpleSettingsDialog(self.app, self.app.settings_manager, self.lang)
        except Exception as e:
            messagebox.showerror(self.lang.get_string("error_title"), self.lang.get_string("settings_error_message").format(e=e))
    
    def on_translation_mode_change(self, *args):
        """Handle changes in the translation mode selection."""
        mode = self.translation_mode_var.get()
        self.create_mode_specific_settings(mode)
        self.set_translation_status(mode)

    def set_translation_status(self, mode):
        """Update related UI elements based on translation mode."""
        # Target language is always available - user can choose whether to translate or not
        self.target_lang_combo.configure(state="readonly")
        
        # Note: LLM translation is completely independent from basic processing output format
        # LLM translation creates separate output files regardless of basic settings
        # The output_format_var only affects the basic Google translation output
        
        # Only update target language if needed, but don't force output format changes
        if mode == "none":
            # No LLM translation - user can choose any basic translation settings
            pass  # Don't modify any basic processing settings
        else:
            # LLM translation modes (local/api) are completely independent
            # They create their own output files, so don't affect basic processing
            pass

    def update_status(self, message_key, **kwargs):
        """Update status bar with a translatable message."""
        message_template = self.lang.get_string(message_key, message_key)
        try:
            formatted_message = message_template.format(**kwargs)
            self.status_var.set(formatted_message)
        except KeyError as e:
            print(f"Status Update Error: Missing key {e} in kwargs for template '{message_template}'")
            self.status_var.set(message_template) # Fallback to template
        self.app.update_idletasks()