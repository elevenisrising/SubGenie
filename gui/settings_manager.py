"""
Settings Manager for SubGenie GUI
================================

Handles application settings persistence and management.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import logging

class SettingsManager:
    """Manages application settings and preferences."""
    
    def __init__(self):
        self.settings_file = Path("settings.json")
        self.default_settings = {
            "appearance": {
                "theme": "system",  # system, dark, light
                "color_theme": "blue"  # blue, green, dark-blue
            },
            "processing": {
                "source_language": "auto",
                "target_language": "zh-CN",
                "whisper_model": "medium",
                "output_format": "bilingual",
                "max_chars": 80,
                "chunk_size": 5
            },
            "translation": {
                "mode": "free",  # free, local, api
                "local_model": "qwen2.5:7b",
                "api_provider": "gemini",
                "api_model": "gemini-2.5-pro"
            },
            "api_settings": {
                "relay_api": {
                    "base_url": "https://www.chataiapi.com/v1/chat/completions",
                    "api_key": "",
                    "default_model": "gemini-2.5-pro"
                },
                "openai": {
                    "api_key": "",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini"
                },
                "anthropic": {
                    "api_key": "",
                    "model": "claude-3-5-sonnet-20241022"
                },
                "deepseek": {
                    "api_key": "",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat"
                }
            },
            "paths": {
                "input_directory": "input_audio",
                "output_directory": "output_subtitles",
                "cache_directory": ".cache",
                "logs_directory": "logs"
            },
            "advanced": {
                "enable_logging": True,
                "log_level": "INFO",
                "auto_save_settings": True,
                "check_updates": True
            }
        }
        self.current_settings = self.default_settings.copy()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    file_settings = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                self.current_settings = self._deep_merge(self.default_settings.copy(), file_settings)
                logging.info(f"Settings loaded from {self.settings_file}")
            else:
                self.current_settings = self.default_settings.copy()
                logging.info("Using default settings")
                
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            self.current_settings = self.default_settings.copy()
        
        return self.current_settings
    
    def save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_settings, f, indent=4, ensure_ascii=False)
            logging.info(f"Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            return False
    
    def get_setting(self, key_path: str, default=None):
        """Get a specific setting using dot notation (e.g., 'processing.source_language')."""
        keys = key_path.split('.')
        value = self.current_settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, key_path: str, value: Any):
        """Set a specific setting using dot notation."""
        keys = key_path.split('.')
        setting_dict = self.current_settings
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in setting_dict:
                setting_dict[key] = {}
            setting_dict = setting_dict[key]
        
        # Set the final value
        setting_dict[keys[-1]] = value
        
        # Auto-save if enabled
        if self.get_setting('advanced.auto_save_settings', True):
            self.save_settings()
    
    def reset_settings(self):
        """Reset all settings to defaults."""
        self.current_settings = self.default_settings.copy()
        self.save_settings()
        logging.info("Settings reset to defaults")
    
    def reset_section(self, section: str):
        """Reset a specific section to defaults."""
        if section in self.default_settings:
            self.current_settings[section] = self.default_settings[section].copy()
            self.save_settings()
            logging.info(f"Settings section '{section}' reset to defaults")
    
    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file."""
        try:
            export_path = Path(file_path)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_settings, f, indent=4, ensure_ascii=False)
            logging.info(f"Settings exported to {export_path}")
            return True
        except Exception as e:
            logging.error(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file."""
        try:
            import_path = Path(file_path)
            if not import_path.exists():
                raise FileNotFoundError(f"Settings file not found: {import_path}")
            
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_settings = json.load(f)
            
            # Validate and merge with defaults
            self.current_settings = self._deep_merge(self.default_settings.copy(), imported_settings)
            self.save_settings()
            logging.info(f"Settings imported from {import_path}")
            return True
        except Exception as e:
            logging.error(f"Error importing settings: {e}")
            return False
    
    def get_api_settings(self, provider: str) -> Dict[str, Any]:
        """Get API settings for a specific provider."""
        return self.get_setting(f'api_settings.{provider}', {})
    
    def set_api_settings(self, provider: str, settings: Dict[str, Any]):
        """Set API settings for a specific provider."""
        self.set_setting(f'api_settings.{provider}', settings)
    
    def is_api_configured(self, provider: str) -> bool:
        """Check if API provider is properly configured."""
        api_settings = self.get_api_settings(provider)
        return bool(api_settings.get('api_key', '').strip())
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get all processing-related settings."""
        return self.get_setting('processing', {})
    
    def get_translation_settings(self) -> Dict[str, Any]:
        """Get all translation-related settings."""
        return self.get_setting('translation', {})
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def validate_settings(self) -> Dict[str, list]:
        """Validate current settings and return any issues."""
        issues = {
            'warnings': [],
            'errors': []
        }
        
        # Validate processing settings
        processing = self.get_processing_settings()
        
        max_chars = processing.get('max_chars', 80)
        if not isinstance(max_chars, int) or max_chars < 10 or max_chars > 200:
            issues['errors'].append("Max characters must be between 10 and 200")
        
        chunk_size = processing.get('chunk_size', 5)
        if not isinstance(chunk_size, int) or chunk_size < 1 or chunk_size > 50:
            issues['errors'].append("Chunk size must be between 1 and 50")
        
        # Validate API settings
        translation = self.get_translation_settings()
        if translation.get('mode') == 'api':
            provider = translation.get('api_provider', '')
            if not self.is_api_configured(provider):
                issues['warnings'].append(f"API provider '{provider}' is not configured")
        
        # Validate paths
        paths = self.get_setting('paths', {})
        for path_name, path_value in paths.items():
            if path_value and not Path(path_value).exists():
                try:
                    Path(path_value).mkdir(parents=True, exist_ok=True)
                    issues['warnings'].append(f"Created missing directory: {path_value}")
                except Exception:
                    issues['errors'].append(f"Cannot create directory: {path_value}")
        
        return issues