"""
Global constants and default values definition.
Unifies parameter defaults across all modules to avoid inconsistency.

IMPORTANT: This is the SINGLE SOURCE OF TRUTH for all default parameters.
All other modules should import and use these constants instead of defining their own defaults.
"""

import os
from pathlib import Path

# Base paths - relative to current working directory
def get_base_dir():
    """Get the current working directory as base directory."""
    return Path.cwd()

def get_input_dir():
    """Get input directory relative to current working directory."""
    return get_base_dir() / "input_audio"

def get_output_dir():
    """Get output directory relative to current working directory."""
    return get_base_dir() / "output_subtitles"

def get_cache_dir():
    """Get cache directory relative to current working directory."""
    return get_base_dir() / ".cache"

def get_logs_dir():
    """Get logs directory relative to current working directory."""
    return get_base_dir() / "logs"

# Directory names (for backward compatibility)
INPUT_DIR_NAME = "input_audio"
OUTPUT_DIR_NAME = "output_subtitles"
CACHE_DIR_NAME = ".cache"
LOGS_DIR_NAME = "logs"

# =============================================================================
# PARAMETER DEFAULTS - SINGLE SOURCE OF TRUTH
# =============================================================================
# Any parameter that needs to be consistent between frontend and backend 
# should be defined here and imported by all modules.

# Whisper model settings
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_SOURCE_LANGUAGE = "auto"
DEFAULT_TARGET_LANGUAGE = "none"  # Default to no translation

# Audio processing settings
DEFAULT_CHUNK_DURATION = 30  # minutes (chunk_time in GUI)
DEFAULT_LONG_AUDIO_THRESHOLD = 15  # minutes
DEFAULT_SEARCH_WINDOW = 120  # seconds
DEFAULT_TARGET_DBFS = -20.0
DEFAULT_DENOISE_STRENGTH = 0.5
DEFAULT_MIN_SPEAKER_DURATION = 2.0  # seconds

# Subtitle formatting settings
DEFAULT_MAX_SUBTITLE_CHARS = 80  # max_chars in GUI and processing
DEFAULT_OUTPUT_FORMAT = "source"  # output_format: source/bilingual/target
DEFAULT_SEGMENTATION_STRATEGY = "spacy"  # spacy/whisper (updated from rule_based)

# Translation settings  
DEFAULT_CHUNK_SIZE = 3  # chunk_size: LLM translation subtitle batch size (GUI shows different default)
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"
DEFAULT_API_MODEL = "gemini-2.5-pro"

# Parallel processing settings
DEFAULT_MAX_WORKERS = 1  # max_workers in GUI and processing (avoid GPU competition)
DEFAULT_PARALLEL_ENABLED = True

# Cache settings
DEFAULT_CACHE_EXPIRY_DAYS = 7

# GUI-specific defaults (might differ from processing defaults for UX reasons)
GUI_DEFAULT_CHUNK_SIZE = 5  # GUI shows 5 by default, but processing can use 3
GUI_DEFAULT_MAX_WORKERS = 1  # Same as processing (avoid GPU competition)
GUI_DEFAULT_CHUNK_TIME = 30  # Same as DEFAULT_CHUNK_DURATION

# Processing constants (these should NOT be changed between GUI and processing)
MERGE_WORD_THRESHOLD = 5
FINAL_PUNCTUATION = '.?!。？！…'
MAX_PUNCTUATION_PER_3_WORDS = 1
MAX_COMMAS_PER_4_WORDS = 1
FUZZY_MATCH_THRESHOLD = 0.8
MIN_SEGMENT_LENGTH = 20
NOISE_REDUCTION_FACTOR = 0.3
SPEAKER_CHANGE_THRESHOLD = 0.5
WINDOW_SIZE_SEC = 0.1

# =============================================================================
# PARAMETER ACCESS HELPERS
# =============================================================================

def get_default_parameters():
    """Get a dictionary of all default parameters for easy access."""
    return {
        # Whisper settings
        'whisper_model': DEFAULT_WHISPER_MODEL,
        'source_language': DEFAULT_SOURCE_LANGUAGE,
        'target_language': DEFAULT_TARGET_LANGUAGE,
        
        # Audio processing
        'chunk_duration': DEFAULT_CHUNK_DURATION,
        'long_audio_threshold': DEFAULT_LONG_AUDIO_THRESHOLD,
        'search_window': DEFAULT_SEARCH_WINDOW,
        'target_dbfs': DEFAULT_TARGET_DBFS,
        'denoise_strength': DEFAULT_DENOISE_STRENGTH,
        'min_speaker_duration': DEFAULT_MIN_SPEAKER_DURATION,
        
        # Subtitle formatting
        'max_subtitle_chars': DEFAULT_MAX_SUBTITLE_CHARS,
        'output_format': DEFAULT_OUTPUT_FORMAT,
        'segmentation_strategy': DEFAULT_SEGMENTATION_STRATEGY,
        
        # Translation
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'local_model': DEFAULT_LOCAL_MODEL,
        'api_model': DEFAULT_API_MODEL,
        
        # Performance
        'max_workers': DEFAULT_MAX_WORKERS,
        'parallel_enabled': DEFAULT_PARALLEL_ENABLED,
        
        # Cache
        'cache_expiry_days': DEFAULT_CACHE_EXPIRY_DAYS,
    }

def get_gui_default_parameters():
    """Get a dictionary of GUI-specific default parameters."""
    defaults = get_default_parameters()
    # Override with GUI-specific values where different
    defaults.update({
        'chunk_size': GUI_DEFAULT_CHUNK_SIZE,
        'max_workers': GUI_DEFAULT_MAX_WORKERS,
        'chunk_time': GUI_DEFAULT_CHUNK_TIME,
    })
    return defaults

def get_processing_constants():
    """Get a dictionary of processing constants that should not be configurable."""
    return {
        'merge_word_threshold': MERGE_WORD_THRESHOLD,
        'final_punctuation': FINAL_PUNCTUATION,
        'max_punctuation_per_3_words': MAX_PUNCTUATION_PER_3_WORDS,
        'max_commas_per_4_words': MAX_COMMAS_PER_4_WORDS,
        'fuzzy_match_threshold': FUZZY_MATCH_THRESHOLD,
        'min_segment_length': MIN_SEGMENT_LENGTH,
        'noise_reduction_factor': NOISE_REDUCTION_FACTOR,
        'speaker_change_threshold': SPEAKER_CHANGE_THRESHOLD,
        'window_size_sec': WINDOW_SIZE_SEC,
    }