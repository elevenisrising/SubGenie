import json
from pathlib import Path

class LanguageManager:
    """Manages loading and switching between language packs."""

    def __init__(self, language_dir="assets/languages", default_language="en"):
        self.language_dir = Path(language_dir)
        self.languages = {}
        self.current_language = default_language
        self._load_languages()
        self.set_language(self.current_language)

    def _load_languages(self):
        """Load all language JSON files from the specified directory."""
        if not self.language_dir.is_dir():
            print(f"Warning: Language directory not found at '{self.language_dir}'")
            return
        for file_path in self.language_dir.glob("*.json"):
            lang_code = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.languages[lang_code] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading language file {file_path}: {e}")

    def set_language(self, lang_code):
        """Set the active language for the application."""
        if lang_code in self.languages:
            self.current_language = lang_code
            self.current_strings = self.languages[lang_code]
            print(f"Language set to: {lang_code}")
        else:
            print(f"Warning: Language '{lang_code}' not found. Using default 'en'.")
            self.current_language = "en"
            self.current_strings = self.languages.get("en", {})

    def get_string(self, key, default=None):
        """Retrieve a string for the given key in the current language."""
        return self.current_strings.get(key, default if default is not None else key)

    def get_available_languages(self):
        """Return a list of available language codes."""
        return list(self.languages.keys())
