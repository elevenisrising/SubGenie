# main.py
import argparse
import datetime
import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import spaCy - make it optional for now
spacy = None
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully")
except ImportError:
    logging.warning("spaCy not available (typer dependency issue), will use fallback segmentation")
    spacy = None
    nlp = None
except OSError:
    logging.warning("spaCy English model 'en_core_web_sm' not found, will use fallback segmentation")
    spacy = None  
    nlp = None

# Import clean segmentation logic
try:
    from .clean_segmentation import structure_and_split_segments
except ImportError:
    # Handle when run as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clean_segmentation import structure_and_split_segments
import numpy as np
import srt
import torch
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent
from tqdm import tqdm

# --- Configuration ---
LOG_FORMAT = '%(asctime)s - [%(levelname)s] - %(message)s'

# Configure logging to write to file AND console
import os
from pathlib import Path

# Ensure logs directory exists
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "timestamp_debug.log"

# Set up file handler
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 'w' to overwrite each run
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Set up console handler (optional, for immediate feedback)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings/errors on console
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Configure root logger  
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Enable DEBUG level for normalization logs
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.info(f"🚀 DEBUG: Timestamp debug logging started - output file: {log_file}")
logging.info("🚀 DEBUG: This session will contain detailed alignment debugging information")

# Import constants
try:
    from src.utils.constants import (
        DEFAULT_CHUNK_DURATION, DEFAULT_LONG_AUDIO_THRESHOLD, DEFAULT_SEARCH_WINDOW,
        DEFAULT_MAX_SUBTITLE_CHARS, DEFAULT_CACHE_EXPIRY_DAYS, DEFAULT_SEGMENTATION_STRATEGY,
        get_processing_constants, get_cache_dir
    )
    # Get processing constants
    PROCESSING_CONSTANTS = get_processing_constants()
except ImportError:
    # Fallback values if constants file not available
    DEFAULT_CHUNK_DURATION = 30
    DEFAULT_LONG_AUDIO_THRESHOLD = 15
    DEFAULT_SEARCH_WINDOW = 120
    DEFAULT_MAX_SUBTITLE_CHARS = 80
    DEFAULT_CACHE_EXPIRY_DAYS = 7
    DEFAULT_SEGMENTATION_STRATEGY = "spacy"
    PROCESSING_CONSTANTS = {
        'merge_word_threshold': 5,
        'final_punctuation': '.?!。？！…',
        'max_punctuation_per_3_words': 1,
        'max_commas_per_4_words': 1,
        'fuzzy_match_threshold': 0.8,
        'min_segment_length': 20,
        'noise_reduction_factor': 0.3,
        'speaker_change_threshold': 0.5,
        'window_size_sec': 0.1,
    }

# --- Constants ---
GLOSSARY_FILE_NAME = "src/utils/glossary.json"

# Use constants from centralized config
try:
    CACHE_DIR = str(get_cache_dir())
except (ImportError, NameError):
    CACHE_DIR = ".cache"

# Extract processing constants for easier access
MERGE_WORD_THRESHOLD = PROCESSING_CONSTANTS['merge_word_threshold']
FINAL_PUNCTUATION = PROCESSING_CONSTANTS['final_punctuation']
MAX_PUNCTUATION_PER_3_WORDS = PROCESSING_CONSTANTS['max_punctuation_per_3_words']
MAX_COMMAS_PER_4_WORDS = PROCESSING_CONSTANTS['max_commas_per_4_words']
FUZZY_MATCH_THRESHOLD = PROCESSING_CONSTANTS['fuzzy_match_threshold']
MIN_SEGMENT_LENGTH = PROCESSING_CONSTANTS['min_segment_length']
NOISE_REDUCTION_FACTOR = PROCESSING_CONSTANTS['noise_reduction_factor']
SPEAKER_CHANGE_THRESHOLD = PROCESSING_CONSTANTS['speaker_change_threshold']
WINDOW_SIZE_SEC = PROCESSING_CONSTANTS['window_size_sec']

# --- Cache Management ---
def get_audio_hash(audio_path: Path) -> str:
    """Calculate SHA256 hash of an audio file for caching."""
    try:
        with open(audio_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "no_hash"

def get_cache_path(audio_hash: str, model_name: str, args_hash: str) -> Path:
    """Generate cache file path using centralized cache directory."""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{audio_hash}_{model_name}_{args_hash}.pkl"

def is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(cache_path.stat().st_mtime)
    return file_age.days < DEFAULT_CACHE_EXPIRY_DAYS

def save_to_cache(cache_path: Path, data: Any):
    """Save data to cache file."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logging.warning(f"Failed to save cache to {cache_path}: {e}")

def load_from_cache(cache_path: Path) -> Any:
    """Load data from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"Failed to load cache from {cache_path}: {e}")
        return None

# --- Glossary and Translation ---
def load_glossary(project_dir: Path) -> Dict:
    """Loads the glossary file from the project directory."""
    glossary_path = project_dir / GLOSSARY_FILE_NAME
    if not glossary_path.exists():
        return {"exact_replace": {}, "pre_translate": {}}
    try:
        with open(glossary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            "exact_replace": {k.lower(): v for k, v in data.get("exact_replace", {}).items()},
            "pre_translate": data.get("pre_translate", {})
        }
    except Exception as e:
        logging.warning(f"Could not load glossary: {e}")
        return {"exact_replace": {}, "pre_translate": {}}

def apply_glossary(text: str, rules: Dict) -> str:
    """Applies exact replacement rules from the glossary."""
    for k, v in rules.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text, flags=re.IGNORECASE)
    return text

def translate_text(text: str, target_lang: str, pre_translate_rules: dict, source_lang='auto') -> str:
    """Translates a single string of text."""
    if text in pre_translate_rules:
        return pre_translate_rules[text]
    if not text.strip() or re.match(r'^[^\w]*$', text):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        logging.warning(f"Translation failed for '{text[:20]}...': {e}")
        return text

def finalize_subtitle_text(text: str) -> str:
    """Removes trailing periods and commas from a subtitle line."""
    return re.sub(r'[,.，。]$', '', text.strip())

# --- Path and Project Management ---
def get_project_paths(input_file_or_project_name: str) -> Dict[str, Path]:
    """Generates all necessary paths from an input file or project name."""
    base_name = Path(input_file_or_project_name).stem
    
    # Use centralized path constants
    try:
        from src.utils.constants import get_input_dir, get_output_dir, get_cache_dir
        input_audio_dir = get_input_dir()
        output_dir = get_output_dir() / base_name
        cache_base_dir = get_cache_dir()
    except ImportError:
        # Fallback to relative paths if constants not available
        input_audio_dir = Path("input_audio")
        output_dir = Path("output_subtitles") / base_name
        cache_base_dir = Path(".cache")
    
    # Check if input is a project name (look in input_audio directory)
    if input_audio_dir.exists():
        # Look for audio files with this name
        possible_files = list(input_audio_dir.glob(f"{base_name}.*"))
        if possible_files:
            # Use the first matching file
            input_file = possible_files[0]
            logging.info(f"Found input file: {input_file}")
        else:
            # Fall back to direct path
            input_file = Path(input_file_or_project_name)
    else:
        # Fall back to direct path
        input_file = Path(input_file_or_project_name)
    
    paths = {
        "output_dir": output_dir,
        "input_file": input_file,
        "extracted_audio": output_dir / f"{base_name}_audio.wav",
        "timing_info": output_dir / "chunk_timing.json",
        "progress_file": output_dir / ".progress.json",
        "raw_chunks_dir": output_dir / "chunks_raw",
        "preprocessed_chunks_dir": output_dir / "chunks_preprocessed",
        "final_srt": output_dir / f"{base_name}_merged.srt",
    }
    
    for key, path in paths.items():
        if 'dir' in key:
            path.mkdir(parents=True, exist_ok=True)
    return paths

def get_chunk_start_time(chunk_index: int, timing_info_path: Path) -> float:
    """Reads the timing JSON and returns the start time in seconds for a specific chunk."""
    if not timing_info_path.exists():
        logging.error(f"Timing info file not found at: {timing_info_path}")
        return 0.0
    try:
        with open(timing_info_path, 'r', encoding='utf-8') as f:
            timings = json.load(f)
        for timing in timings:
            if timing['chunk_index'] == chunk_index:
                return timing['start_time_ms'] / 1000.0
        logging.warning(f"Chunk index {chunk_index} not found in timing file.")
        return 0.0
    except Exception as e:
        logging.error(f"Failed to read timing info: {e}")
        return 0.0

# --- Audio Preprocessing ---
def normalize_audio_volume(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize audio volume to target dBFS."""
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs)

def denoise_audio(audio: AudioSegment, strength: float = 0.5, use_ai_separation: bool = False) -> AudioSegment:
    """Apply noise reduction to audio while preserving low-volume speech."""
    
    if use_ai_separation:
        return extract_vocals_with_spleeter(audio, strength)
    else:
        return apply_traditional_denoise(audio, strength)


def apply_traditional_denoise(audio: AudioSegment, strength: float) -> AudioSegment:
    """Traditional gentle noise reduction method."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Use a higher percentile for noise floor to be more conservative
    noise_floor = np.percentile(np.abs(samples), 5)  # Changed from 10 to 5
    
    # Use a more conservative threshold calculation
    threshold = noise_floor * (1 + strength * 0.5)  # Reduced multiplier effect
    
    # Apply gentle reduction instead of hard cutoff
    # Reduce amplitude instead of zeroing out
    mask = np.abs(samples) < threshold
    denoised_samples = samples.copy()
    denoised_samples[mask] = samples[mask] * NOISE_REDUCTION_FACTOR
    
    # Ensure we don't clip
    denoised_samples = np.clip(denoised_samples, -32767, 32767).astype(np.int16)
    
    return audio._spawn(denoised_samples)


def extract_vocals_with_spleeter(audio: AudioSegment, mix_ratio: float) -> AudioSegment:
    """Extract vocals using AI separation (Spleeter)."""
    try:
        import tempfile
        import os
        
        # Check if spleeter is available
        try:
            from spleeter.separator import Separator
            import librosa
            import soundfile as sf
        except ImportError:
            logging.warning("Spleeter not installed, falling back to traditional denoising")
            return apply_traditional_denoise(audio, mix_ratio)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            temp_vocal_path = temp_input.name.replace('.wav', '_vocals.wav')
            
            # Export original audio to temp file
            audio.export(temp_input.name, format='wav')
            
            # Initialize separator
            separator = Separator('spleeter:2stems-16kHz')
            
            # Load and process audio
            waveform, sr = librosa.load(temp_input.name, sr=16000, mono=False)
            
            # Ensure 2D array for spleeter
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            
            # Separate vocals and accompaniment
            prediction = separator.separate(waveform)
            
            # Extract vocals
            vocals = prediction['vocals']
            
            # Convert to mono if stereo
            if vocals.shape[0] > 1:
                vocals_mono = np.mean(vocals, axis=0)
            else:
                vocals_mono = vocals[0]
            
            # Resample to original sample rate if needed
            if sr != audio.frame_rate:
                vocals_resampled = librosa.resample(vocals_mono, orig_sr=sr, target_sr=audio.frame_rate)
            else:
                vocals_resampled = vocals_mono
            
            # Convert to AudioSegment format
            vocals_int16 = np.clip(vocals_resampled * 32767, -32767, 32767).astype(np.int16)
            vocals_audio = audio._spawn(vocals_int16)
            
            # Clean up temp files
            os.unlink(temp_input.name)
            if os.path.exists(temp_vocal_path):
                os.unlink(temp_vocal_path)
            
            # Mix original and extracted vocals based on mix_ratio
            # mix_ratio closer to 1.0 = more vocals, less background
            original_weight = max(0.1, 1.0 - mix_ratio)  # Keep at least 10% original
            vocal_weight = mix_ratio
            
            # Ensure same length - pad shorter audio if needed
            original_length = len(audio)
            vocals_length = len(vocals_audio)
            
            if vocals_length < original_length:
                # Pad vocals_audio to match original length
                padding = AudioSegment.silent(duration=original_length - vocals_length)
                vocals_audio = vocals_audio + padding
                logging.warning(f"AI separation result was shorter ({vocals_length}ms vs {original_length}ms), padded with silence")
            elif vocals_length > original_length:
                # Trim vocals_audio to match original length
                vocals_audio = vocals_audio[:original_length]
                logging.warning(f"AI separation result was longer ({vocals_length}ms vs {original_length}ms), trimmed to match")
            
            mixed = (audio * original_weight + vocals_audio * vocal_weight)
            
            logging.info(f"AI vocal separation completed (mix ratio: {mix_ratio:.1f})")
            return mixed
            
    except Exception as e:
        logging.warning(f"AI vocal separation failed: {e}, falling back to traditional denoising")
        return apply_traditional_denoise(audio, mix_ratio)

def detect_speaker_changes(audio: AudioSegment, min_duration_sec: float) -> List[float]:
    """Detect potential speaker changes in audio."""
    samples = np.array(audio.get_array_of_samples())
    frame_rate = audio.frame_rate
    window_size = int(WINDOW_SIZE_SEC * frame_rate)  # 100ms windows
    
    features = []
    for i in range(0, len(samples) - window_size, window_size):
        window = samples[i:i+window_size].astype(np.float64)
        if len(window) > 0:
            rms = np.sqrt(np.mean(window**2))
            features.append(rms)
    
    changes = []
    for i in range(1, len(features)):
        if features[i-1] > 0:  # Avoid division by zero
            relative_change = abs(features[i] - features[i-1]) / (features[i-1] + 1e-6)
            if relative_change > SPEAKER_CHANGE_THRESHOLD:
                timestamp = (i * window_size) / frame_rate
                if not changes or timestamp - changes[-1] >= min_duration_sec:
                    changes.append(timestamp)
    
    return changes

# --- Audio Splitting ---
def find_best_split_point(audio: AudioSegment, target_ms: int, window_ms: int) -> int:
    """Find the best point to split audio within a search window."""
    start_search = max(0, target_ms - window_ms // 2)
    end_search = min(len(audio), target_ms + window_ms // 2)
    
    if start_search >= end_search:
        return target_ms
    
    search_area = audio[start_search:end_search]
    
    # Find silent periods
    try:
        silences = detect_nonsilent(search_area, min_silence_len=500, silence_thresh=-40, seek_step=1)
    except:
        return target_ms
    
    if not silences:
        return target_ms
    
    # Find the best gap between non-silent segments
    best_split = target_ms - start_search
    max_gap = 0
    
    if len(silences) > 1:
        for i in range(len(silences) - 1):
            gap_start = silences[i][1]
            gap_end = silences[i+1][0]
            gap_size = gap_end - gap_start
            if gap_size > max_gap:
                max_gap = gap_size
                best_split = gap_start + gap_size // 2
    
    if max_gap > 0:
        return start_search + best_split
    else:
        # Use the middle of the longest silence
        longest_silence = max(silences, key=lambda s: s[1] - s[0])
        silence_middle = longest_silence[0] + (longest_silence[1] - longest_silence[0]) // 2
        return start_search + silence_middle

def plan_and_split_audio(audio_path: Path, paths: Dict[str, Path], args: argparse.Namespace) -> List[Dict]:
    """Split audio into chunks and create timing information with comprehensive validation."""
    logging.info("Loading audio for splitting...")
    audio = AudioSegment.from_file(audio_path)
    
    # Comprehensive duration validation and logging
    duration_ms = len(audio)
    duration_sec = duration_ms / 1000.0
    duration_min = duration_sec / 60.0
    
    logging.info(f"📊 AUDIO DURATION ANALYSIS:")
    logging.info(f"   Raw AudioSegment length: {duration_ms} ms")
    logging.info(f"   Calculated seconds: {duration_sec:.2f} s")
    logging.info(f"   Calculated minutes: {duration_min:.2f} min")
    logging.info(f"   Hours equivalent: {duration_min/60:.2f} hours")
    logging.info(f"   Threshold for chunking: {args.long_audio_threshold} min")
    
    # Validate that duration makes sense
    if duration_sec <= 0:
        raise ValueError(f"Invalid audio duration: {duration_sec} seconds")
    if duration_sec > 24 * 3600:  # More than 24 hours
        logging.warning(f"Unusually long audio detected: {duration_min/60:.2f} hours")
    
    # Create timing information
    if duration_min < args.long_audio_threshold:
        logging.info("Short audio detected. Processing as a single chunk.")
        chunk_path = paths["raw_chunks_dir"] / "chunk_1.wav"
        audio.export(chunk_path, format="wav")
        timing_info = [{
            "chunk_index": 1,
            "start_time_ms": 0,
            "end_time_ms": len(audio),
            "duration_ms": len(audio)
        }]
    else:
        logging.info("Long audio detected. Splitting into manageable chunks...")
        target_len_ms = args.chunk_duration * 60 * 1000
        search_window_ms = args.search_window * 1000
        
        timing_info = []
        start_ms = 0
        chunk_index = 1
        
        # pbar = tqdm(total=len(audio), desc="Splitting audio", unit="ms")
        logging.info(f"Splitting audio of total duration {len(audio)} ms...")
        
        while start_ms < len(audio):
            target_split_ms = start_ms + target_len_ms
            
            if target_split_ms >= len(audio):
                end_ms = len(audio)
            else:
                end_ms = find_best_split_point(audio, target_split_ms, search_window_ms)
            
            # Extract and save chunk
            chunk_audio = audio[start_ms:end_ms]
            chunk_path = paths["raw_chunks_dir"] / f"chunk_{chunk_index}.wav"
            chunk_audio.export(chunk_path, format="wav")
            
            # Record timing information
            timing_info.append({
                "chunk_index": chunk_index,
                "start_time_ms": start_ms,
                "end_time_ms": end_ms,
                "duration_ms": end_ms - start_ms
            })
            
            # pbar.update(end_ms - start_ms)
            chunk_duration_sec = (end_ms - start_ms) / 1000.0
            logging.info(f"Split chunk {chunk_index}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s (duration: {chunk_duration_sec:.2f}s)")
            start_ms = end_ms
            chunk_index += 1
        
        # pbar.close()
    
    # Save timing information
    with open(paths["timing_info"], 'w', encoding='utf-8') as f:
        json.dump(timing_info, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Audio split into {len(timing_info)} chunks.")
    logging.info(f"Timing information saved to: {paths['timing_info']}")
    
    return timing_info

# --- Punctuation-First Segmentation Architecture ---

def apply_intelligent_punctuation(all_words: List[Dict], segment_boundaries: List[Dict], strategy: str) -> Tuple[str, List[Dict], str]:
    """Apply intelligent punctuation strategy and return text + words for mapping.
    
    Args:
        all_words: All Whisper words with timestamps
        segment_boundaries: Segment boundary information for BERT processing
        strategy: Punctuation strategy
        
    Returns:
        tuple: (final_punctuated_text, clean_words_for_mapping, clean_reference_text)
        All strategies use the same clean words/text for mapping, only final text differs
    """
    
    # Step 1: Create unified clean reference for all strategies (no punctuation)
    clean_words_for_mapping = create_clean_words_for_mapping(all_words)
    clean_reference_text = " ".join([w["word"] for w in clean_words_for_mapping])
    
    # Step 2: Generate punctuated text based on strategy
    if strategy == "spacy":
        # Use Whisper's original punctuation, will be processed by spaCy for sentence segmentation
        final_text = " ".join([word["word"].strip() for word in all_words if word["word"].strip()])
        logging.info(f"Strategy: spaCy - Using Whisper punctuation for spaCy segmentation ({len(final_text)} chars)")
        
    elif strategy == "whisper":
        # Use Whisper's original segments directly (bypass spaCy)
        final_text = " ".join([word["word"].strip() for word in all_words if word["word"].strip()])
        logging.info(f"Strategy: Whisper - Using Whisper segments directly ({len(final_text)} chars)")
        
    else:
        # Fallback to spaCy strategy
        final_text = " ".join([word["word"].strip() for word in all_words if word["word"].strip()])
        logging.warning(f"Unknown strategy '{strategy}', falling back to spaCy")
    
    # Return: punctuated text, clean words for mapping, clean text for mapping
    return final_text, clean_words_for_mapping, clean_reference_text

def clean_text_for_bert(text: str) -> str:
    """Remove punctuation from text while preserving contractions, with robust normalization."""
    if not text or not text.strip():
        return ""
    
    original_text = text
    logging.info(f"🧹 DEBUG: Text cleaning - Original: '{original_text[:100]}{'...' if len(original_text) > 100 else ''}'")
    
    # Step 1: Normalize different quote/apostrophe types to standard ASCII
    text = text.replace(''', "'").replace(''', "'")  # Smart quotes to ASCII
    text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
    text = text.replace('—', '-').replace('–', '-')  # Em/en dash to hyphen
    logging.info(f"🧹 DEBUG: After quote normalization: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    # Step 2: Remove punctuation but keep apostrophes and hyphens for contractions
    cleaned = re.sub(r'[^\w\s\'-]', '', text)
    logging.info(f"🧹 DEBUG: After punctuation removal: '{cleaned[:100]}{'...' if len(cleaned) > 100 else ''}'")
    
    # Step 3: Normalize whitespace (multiple spaces to single space)
    cleaned = ' '.join(cleaned.split())
    
    # Step 4: Handle edge cases
    cleaned = cleaned.strip()
    final_text = cleaned
    
    logging.info(f"🧹 DEBUG: Final cleaned text: '{final_text[:100]}{'...' if len(final_text) > 100 else ''}'")
    if original_text.strip() != final_text:
        logging.info(f"🧹 DEBUG: Text changed during cleaning: {len(original_text)} -> {len(final_text)} chars")
    
    return final_text

def create_clean_words_for_mapping(segment_words: List[Dict]) -> List[Dict]:
    """Create cleaned version of words (no punctuation) for timestamp mapping."""
    if not segment_words:
        return []
        
    cleaned_words = []
    
    for word_info in segment_words:
        if not isinstance(word_info, dict) or "word" not in word_info:
            continue
            
        word = word_info["word"].strip()
        if word:
            # Remove punctuation but keep contractions
            clean_word = re.sub(r'[^\w\s\'-]', '', word).strip()
            if clean_word:  # Only add if something remains after cleaning
                cleaned_word_info = word_info.copy()
                cleaned_word_info["word"] = clean_word
                cleaned_words.append(cleaned_word_info)
    
    return cleaned_words

def is_bert_segment_reasonable(text: str, word_count: int) -> bool:
    """Check if BERT output for a segment has reasonable punctuation distribution."""
    if not text or not text.strip() or word_count <= 0:
        return False
    
    # Count punctuation marks
    comma_count = text.count(',')
    period_count = text.count('.')
    question_count = text.count('?')
    exclamation_count = text.count('!')
    
    total_punctuation = comma_count + period_count + question_count + exclamation_count
    
    # Reasonable thresholds based on word count using constants
    max_reasonable_punctuation = max(word_count // 3, MAX_PUNCTUATION_PER_3_WORDS)
    
    if total_punctuation > max_reasonable_punctuation:
        logging.debug(f"BERT segment has too much punctuation: {total_punctuation} marks for {word_count} words")
        return False
    
    # Check for excessive commas specifically using constants
    max_reasonable_commas = max(word_count // 4, MAX_COMMAS_PER_4_WORDS)
    if comma_count > max_reasonable_commas:
        logging.debug(f"BERT segment has too many commas: {comma_count} commas for {word_count} words")
        return False
    
    return True

def map_sentences_to_word_timestamps(sentences: List[str], clean_words_for_mapping: List[Dict], clean_reference_text: str) -> List[Dict]:
    """Map spaCy sentences to Whisper word timestamps using unified clean reference.
    
    Args:
        sentences: spaCy detected sentences with punctuation (final output)
        clean_words_for_mapping: Clean words (no punctuation) with timestamps
        clean_reference_text: Clean reference text (no punctuation) for matching
        
    Returns:
        List of segments with text and precise word-level timestamps
    """
    if not sentences or not clean_words_for_mapping or not clean_reference_text.strip():
        return []
    
    logging.debug(f"Mapping {len(sentences)} sentences to {len(clean_words_for_mapping)} clean words")
    logging.debug(f"Clean reference text: '{clean_reference_text[:100]}...'")
    
    final_segments = []
    used_positions = set()
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        # Remove punctuation from sentence for matching against clean reference
        sentence_for_matching = clean_text_for_bert(sentence)
        
        
        # Find position in clean reference text
        start_pos = find_available_position(sentence_for_matching, clean_reference_text, used_positions)
        
        if start_pos == -1:
            logging.warning(f"Could not find sentence in clean reference: '{sentence_for_matching}'")
            continue
        
        end_pos = start_pos + len(sentence_for_matching) - 1
        
        # Map character positions to word indices
        start_word_idx, end_word_idx = map_char_positions_to_words(
            start_pos, end_pos, clean_words_for_mapping, clean_reference_text
        )
        
        if start_word_idx == -1 or end_word_idx == -1:
            logging.warning(f"📍 DEBUG: Could not map character positions to words for: '{sentence_for_matching}'")
            logging.warning(f"📍 DEBUG: Character positions - start: {start_pos}, end: {end_pos}")
            logging.warning(f"📍 DEBUG: Word mapping result - start_idx: {start_word_idx}, end_idx: {end_word_idx}")
            logging.warning(f"📍 DEBUG: Total clean words available: {len(clean_words_for_mapping)}")
            if clean_words_for_mapping:
                logging.warning(f"📍 DEBUG: First few words: {[w.get('word', '') for w in clean_words_for_mapping[:5]]}")
            continue
        
        # Mark positions as used
        used_positions.update(range(start_pos, end_pos + 1))
        
        # Create segment with precise timing (keep original punctuated sentence)
        final_segments.append({
            'text': sentence,  # Keep original sentence with punctuation
            'start': clean_words_for_mapping[start_word_idx]['start'],
            'end': clean_words_for_mapping[end_word_idx]['end'],
            'word_count': end_word_idx - start_word_idx + 1
        })
        
    
    return final_segments

def find_available_position(target_text: str, reference_text: str, used_positions: set) -> int:
    """Find next available position for target_text in reference_text with intelligent matching."""
    if not target_text or not reference_text:
        logging.warning(f"🔍 DEBUG: find_available_position called with empty text - target: '{target_text}', reference length: {len(reference_text) if reference_text else 0}")
        return -1
    
    logging.info(f"🔍 DEBUG: Finding position for target: '{target_text[:50]}{'...' if len(target_text) > 50 else ''}'")
    logging.info(f"🔍 DEBUG: Reference text length: {len(reference_text)}, used positions: {len(used_positions)}")
    
    # Normalize both texts for comparison
    target_normalized = clean_text_for_bert(target_text).strip()
    reference_normalized = clean_text_for_bert(reference_text).strip()
    
    logging.info(f"🔍 DEBUG: Target normalized: '{target_normalized[:50]}{'...' if len(target_normalized) > 50 else ''}'")
    
    if not target_normalized:
        logging.warning(f"🔍 DEBUG: Target text became empty after normalization")
        return -1
    
    # Method 1: Direct substring search (simple and reliable)
    search_start = 0
    attempts = 0
    while search_start < len(reference_normalized) and attempts < 10:
        attempts += 1
        pos = reference_normalized.find(target_normalized, search_start)
        if pos == -1:
            logging.warning(f"🔍 DEBUG: Target not found after position {search_start}, attempt {attempts}")
            break
        
        logging.info(f"🔍 DEBUG: Found potential match at position {pos}, attempt {attempts}")
        
        # Check if this position range is available
        target_range = set(range(pos, pos + len(target_normalized)))
        overlap = target_range.intersection(used_positions)
        if not overlap:
            logging.info(f"🔍 DEBUG: Position {pos} is available, target length: {len(target_normalized)}")
            return pos
        else:
            logging.info(f"🔍 DEBUG: Position {pos} conflicts with used positions: {list(overlap)[:10]}...")
        
        search_start = pos + 1
    
    # Method 2: Word boundary matching (for partial matches)
    exact_pos = find_word_boundary_match(target_normalized, reference_normalized, used_positions)
    if exact_pos != -1:
        return exact_pos
    
    # Method 3: Token-by-token matching (last resort)
    token_pos = find_token_match(target_normalized, reference_normalized, used_positions)
    if token_pos != -1:
        return token_pos
    return -1

def find_word_boundary_match(target: str, reference: str, used_positions: set) -> int:
    """Find exact match at word boundaries."""
    import re
    
    # Escape special regex characters in target
    escaped_target = re.escape(target)
    
    # Create word boundary pattern
    pattern = r'\b' + escaped_target + r'\b'
    
    search_start = 0
    while search_start < len(reference):
        match = re.search(pattern, reference[search_start:])
        if not match:
            break
        
        actual_pos = search_start + match.start()
        target_range = set(range(actual_pos, actual_pos + len(target)))
        
        # Check if position is available
        if not target_range.intersection(used_positions):
            return actual_pos
        
        search_start = actual_pos + 1
    
    return -1

def find_fuzzy_match(target: str, reference: str, used_positions: set, threshold: float = 0.8) -> int:
    """Find position using fuzzy string matching."""
    target_words = target.split()
    if not target_words:
        return -1
    
    # Try sliding window of same length as target
    reference_words = reference.split()
    target_len = len(target_words)
    
    for start_idx in range(len(reference_words) - target_len + 1):
        window_words = reference_words[start_idx:start_idx + target_len]
        window_text = ' '.join(window_words)
        
        # Calculate character position
        char_pos = len(' '.join(reference_words[:start_idx]))
        if start_idx > 0:
            char_pos += 1  # Add space before
        
        # Check if position is available
        target_range = set(range(char_pos, char_pos + len(window_text)))
        if target_range.intersection(used_positions):
            continue
        
        # Calculate similarity
        similarity = calculate_word_similarity(target_words, window_words)
        if similarity >= threshold:
            return char_pos
    
    return -1

def find_token_match(target: str, reference: str, used_positions: set) -> int:
    """Find match by comparing individual tokens with partial matching support."""
    target_tokens = target.split()
    reference_tokens = reference.split()
    
    if not target_tokens:
        return -1
    
    # Try to find sequence of tokens that match exactly
    for ref_start in range(len(reference_tokens) - len(target_tokens) + 1):
        ref_window = reference_tokens[ref_start:ref_start + len(target_tokens)]
        
        # Calculate character position
        char_pos = len(' '.join(reference_tokens[:ref_start]))
        if ref_start > 0:
            char_pos += 1
        
        window_text = ' '.join(ref_window)
        target_range = set(range(char_pos, char_pos + len(window_text)))
        
        if target_range.intersection(used_positions):
            continue
        
        # Check if tokens match (case-insensitive)
        if [t.lower() for t in target_tokens] == [r.lower() for r in ref_window]:
            return char_pos
    
    # If exact match fails, try partial matching (at least 70% of words)
    min_match_ratio = 0.7
    best_match_pos = -1
    best_match_score = 0
    
    for ref_start in range(len(reference_tokens) - 1):  # Allow shorter windows
        max_window_size = min(len(target_tokens) + 2, len(reference_tokens) - ref_start)
        
        for window_size in range(max(1, len(target_tokens) - 2), max_window_size + 1):
            ref_window = reference_tokens[ref_start:ref_start + window_size]
            
            if not ref_window:
                continue
            
            # Calculate character position
            char_pos = len(' '.join(reference_tokens[:ref_start]))
            if ref_start > 0:
                char_pos += 1
            
            window_text = ' '.join(ref_window)
            target_range = set(range(char_pos, char_pos + len(window_text)))
            
            if target_range.intersection(used_positions):
                continue
            
            # Calculate match score
            target_set = set(t.lower() for t in target_tokens)
            ref_set = set(r.lower() for r in ref_window)
            
            if target_set and ref_set:
                intersection = target_set.intersection(ref_set)
                score = len(intersection) / len(target_set)
                
                if score >= min_match_ratio and score > best_match_score:
                    best_match_score = score
                    best_match_pos = char_pos
    
    return best_match_pos

def calculate_word_similarity(words1: list, words2: list) -> float:
    """Calculate similarity between two word lists."""
    if not words1 or not words2:
        return 0.0
    
    if len(words1) != len(words2):
        return 0.0
    
    matches = sum(1 for w1, w2 in zip(words1, words2) if w1.lower() == w2.lower())
    return matches / len(words1)

def map_char_positions_to_words(start_pos: int, end_pos: int, clean_words: List[Dict], reference_text: str) -> Tuple[int, int]:
    """Map character positions in reference text to word indices with robust handling."""
    if not clean_words or not reference_text:
        return -1, -1
    
    
    # Build robust character to word mapping
    char_to_word_map = build_char_to_word_mapping(clean_words, reference_text)
    
    if not char_to_word_map:
        logging.warning("Failed to build character to word mapping")
        return -1, -1
    
    # Find word indices for start and end positions
    start_word_idx = find_word_at_position(start_pos, char_to_word_map, reference_text)
    end_word_idx = find_word_at_position(end_pos, char_to_word_map, reference_text)
    
    
    # Validate and fix the mapping
    if start_word_idx != -1 and end_word_idx != -1:
        # Ensure end_word_idx >= start_word_idx
        if end_word_idx < start_word_idx:
            logging.info(f"  🔧 Fixed end_word_idx: {end_word_idx} -> {start_word_idx}")
            end_word_idx = start_word_idx
        
        # Validate against clean_words bounds
        original_start = start_word_idx
        original_end = end_word_idx
        start_word_idx = max(0, min(start_word_idx, len(clean_words) - 1))
        end_word_idx = max(0, min(end_word_idx, len(clean_words) - 1))
        
        if start_word_idx != original_start or end_word_idx != original_end:
            pass  # Bounds were corrected
        
        return start_word_idx, end_word_idx
    
    # Fallback: try word-level mapping
    return fallback_word_mapping(start_pos, end_pos, clean_words, reference_text)

def build_char_to_word_mapping(clean_words: List[Dict], reference_text: str) -> Dict[int, int]:
    """Build a robust character to word index mapping."""
    char_to_word_map = {}
    current_pos = 0
    
    for word_idx, word_info in enumerate(clean_words):
        word = word_info["word"].strip()
        if not word:
            continue
        
        # Find word position with some tolerance for whitespace
        word_start_pos = find_word_in_text(word, reference_text, current_pos)
        
        if word_start_pos != -1:
            # Map each character in this word to word_idx
            for char_pos in range(word_start_pos, word_start_pos + len(word)):
                if char_pos < len(reference_text):
                    char_to_word_map[char_pos] = word_idx
            
            # Update position for next search
            current_pos = word_start_pos + len(word)
            
            # Map space after word (if exists)
            if current_pos < len(reference_text) and reference_text[current_pos] == ' ':
                char_to_word_map[current_pos] = word_idx
                current_pos += 1
        else:
            logging.debug(f"Could not find word '{word}' in reference text at position {current_pos}")
    
    return char_to_word_map

def find_word_in_text(word: str, text: str, start_pos: int) -> int:
    """Find word in text with tolerance for whitespace differences."""
    # Try exact match first
    pos = text.find(word, start_pos)
    if pos != -1:
        return pos
    
    # Try with normalized whitespace
    remaining_text = text[start_pos:].lstrip()
    offset = len(text[start_pos:]) - len(remaining_text)
    pos = remaining_text.find(word)
    
    if pos != -1:
        return start_pos + offset + pos
    
    return -1

def find_word_at_position(pos: int, char_to_word_map: Dict[int, int], reference_text: str) -> int:
    """Find word index at given character position with fallback strategies."""
    # Strategy 1: Direct mapping
    if pos in char_to_word_map:
        return char_to_word_map[pos]
    
    # Strategy 2: Search nearby positions (within 3 characters)
    for offset in range(1, 4):
        # Check positions before
        if pos - offset >= 0 and (pos - offset) in char_to_word_map:
            return char_to_word_map[pos - offset]
        
        # Check positions after
        if pos + offset < len(reference_text) and (pos + offset) in char_to_word_map:
            return char_to_word_map[pos + offset]
    
    # Strategy 3: Find closest mapped position
    mapped_positions = sorted(char_to_word_map.keys())
    if not mapped_positions:
        return -1
    
    # Binary search for closest position
    closest_pos = min(mapped_positions, key=lambda x: abs(x - pos))
    return char_to_word_map[closest_pos]

def fallback_word_mapping(start_pos: int, end_pos: int, clean_words: List[Dict], reference_text: str) -> Tuple[int, int]:
    """Fallback word mapping when character mapping fails."""
    
    # Extract target text
    if start_pos >= 0 and end_pos < len(reference_text) and end_pos >= start_pos:
        target_text = reference_text[start_pos:end_pos + 1].strip()
        target_words = target_text.split()
        
        
        if target_words:
            # Find word sequence in clean_words
            clean_word_texts = [w["word"] for w in clean_words]
            
            
            # Try to find matching sequence
            for start_idx in range(len(clean_word_texts) - len(target_words) + 1):
                window = clean_word_texts[start_idx:start_idx + len(target_words)]
                
                # Case-insensitive comparison
                if [w.lower() for w in target_words] == [w.lower() for w in window]:
                    return start_idx, start_idx + len(target_words) - 1
            
            pass  # No exact word sequence match found
        else:
            pass  # Target text has no words
    else:
        pass  # Invalid position bounds
    
    # Last resort: return reasonable bounds
    mid_word_idx = len(clean_words) // 2
    return max(0, mid_word_idx - 1), min(len(clean_words) - 1, mid_word_idx + 1)

def find_sentence_in_word_sequence(sentence: str, word_sequence: List[str], word_positions: List[Dict], 
                                   final_text: str, used_indices: set) -> Tuple[int, int]:
    """Find sentence boundaries in word sequence using sliding window approach."""
    if not sentence or not word_sequence:
        return -1, -1
    
    # Extract words from sentence (handle mixed punctuation scenarios)
    sentence_words = []
    
    # Try multiple extraction methods for robustness
    # Method 1: Extract all word-like tokens
    word_pattern = r'\b\w+(?:\'\w+)?\b'  # Matches words and contractions
    matches = re.finditer(word_pattern, sentence)
    sentence_words = [match.group().lower() for match in matches]
    
    if not sentence_words:
        logging.debug(f"No words extracted from sentence: '{sentence}'")
        return -1, -1
    
    # Sliding window search in word_sequence
    for start_idx in range(len(word_sequence) - len(sentence_words) + 1):
        # Skip if this range overlaps with used indices
        if any(idx in used_indices for idx in range(start_idx, start_idx + len(sentence_words))):
            continue
        
        # Check if this window matches our sentence words
        window = word_sequence[start_idx:start_idx + len(sentence_words)]
        
        if window == sentence_words:
            end_idx = start_idx + len(sentence_words) - 1
            logging.debug(f"Found exact match at word indices {start_idx}-{end_idx}")
            return start_idx, end_idx
    
    # Fallback: fuzzy matching with tolerance for small differences
    for start_idx in range(len(word_sequence) - len(sentence_words) + 1):
        if any(idx in used_indices for idx in range(start_idx, start_idx + len(sentence_words))):
            continue
        
        window = word_sequence[start_idx:start_idx + len(sentence_words)]
        
        # Calculate similarity (allow for small differences)
        matches = sum(1 for i, word in enumerate(sentence_words) if i < len(window) and window[i] == word)
        similarity = matches / len(sentence_words)
        
        if similarity >= FUZZY_MATCH_THRESHOLD:
            end_idx = start_idx + len(sentence_words) - 1
            logging.debug(f"Found fuzzy match ({similarity:.2f}) at word indices {start_idx}-{end_idx}")
            return start_idx, end_idx
    
    return -1, -1

def split_long_segment(segment: Dict, max_chars: int, original_words: List[Dict] = None) -> List[Dict]:
    """Split overly long segments intelligently with PRECISE word-level timing."""
    text = segment['text']
    start_time = segment['start']
    end_time = segment['end']
    
    if len(text) <= max_chars:
        return [segment]
    
    logging.debug(f"Splitting long segment ({len(text)} chars): {text[:50]}...")
    
    # Strategy 1: Split at conjunctions
    conjunction_splits = split_at_conjunctions(text, max_chars)
    if len(conjunction_splits) > 1:
        return _create_split_segments_with_timing(conjunction_splits, segment, original_words)
    
    # Strategy 2: Split at punctuation
    punctuation_splits = split_at_punctuation(text, max_chars)
    if len(punctuation_splits) > 1:
        return _create_split_segments_with_timing(punctuation_splits, segment, original_words)
    
    # Strategy 2.5: Intelligent semantic breakpoint detection
    semantic_splits = split_at_semantic_breakpoints(text, max_chars)
    if len(semantic_splits) > 1:
        return _create_split_segments_with_timing(semantic_splits, segment, original_words)
    
    # Strategy 3: Force split at word boundaries (last resort)
    word_splits = force_split_at_words(text, max_chars)
    return _create_split_segments_with_timing(word_splits, segment, original_words)

def _create_split_segments_with_timing(text_splits: List[str], original_segment: Dict, 
                                      original_words: List[Dict] = None) -> List[Dict]:
    """Create split segments using precise word-level timing when available."""
    if not text_splits:
        return []
    
    # If we have word-level data, use precise timing
    if original_words and len(original_words) > 0:
        try:
            # Find the word range that corresponds to this segment
            segment_words = _extract_segment_words(original_segment, original_words)
            if segment_words:
                return create_precise_timed_segments(text_splits, segment_words)
        except Exception as e:
            logging.warning(f"Failed to use precise timing, falling back to proportional: {e}")
    
    # Fallback to proportional timing
    return create_timed_segments(text_splits, original_segment['start'], original_segment['end'])

def _extract_segment_words(segment: Dict, all_words: List[Dict]) -> List[Dict]:
    """Extract the Whisper words that correspond to this segment's timespan."""
    if not all_words:
        return []
    
    segment_start = segment['start']
    segment_end = segment['end']
    
    # Find words that fall within this segment's time range (with small tolerance)
    tolerance = 0.1  # 100ms tolerance for timing precision
    segment_words = []
    
    for word_data in all_words:
        word_start = word_data.get('start', 0.0)
        word_end = word_data.get('end', 0.0)
        
        # Word overlaps with segment if there's any time intersection
        if (word_end >= segment_start - tolerance and 
            word_start <= segment_end + tolerance):
            segment_words.append(word_data)
    
    return segment_words

def split_at_conjunctions(text: str, max_chars: int) -> List[str]:
    """Split text at conjunction points."""
    if not text or not text.strip():
        return []
    
    conjunctions = r'\b(because|which|but|and|so|then|however|therefore|meanwhile|while|although|unless|since)\b'
    
    # Find all conjunction positions
    matches = list(re.finditer(conjunctions, text, re.IGNORECASE))
    
    if not matches:
        return [text]  # No conjunctions found
    
    # Find optimal split points
    splits = []
    current_pos = 0
    
    for match in matches:
        split_pos = match.start()
        
        # Check if splitting here would create reasonable segments
        part_before = text[current_pos:split_pos].strip()
        remaining = text[split_pos:].strip()
        
        if (len(part_before) >= MIN_SEGMENT_LENGTH and len(part_before) <= max_chars and 
            len(remaining) > MIN_SEGMENT_LENGTH):  # Don't create too small segments
            splits.append(part_before)
            current_pos = split_pos
    
    # Add remaining text
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            splits.append(remaining)
    
    return splits if len(splits) > 1 else [text]

def split_at_punctuation(text: str, max_chars: int) -> List[str]:
    """Split text at punctuation marks."""
    if not text or not text.strip():
        return []
    
    # Split at sentence-ending punctuation
    sentences = re.split(r'([.!?;:])', text)
    
    if len(sentences) <= 2:  # No meaningful split
        return [text]
    
    splits = []
    current_segment = ""
    
    for i in range(0, len(sentences), 2):
        sentence_part = sentences[i] if i < len(sentences) else ""
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_part = sentence_part + punct
        
        if len(current_segment + full_part) <= max_chars:
            current_segment += full_part
        else:
            if current_segment.strip():
                splits.append(current_segment.strip())
            current_segment = full_part
    
    if current_segment.strip():
        splits.append(current_segment.strip())
    
    return splits if len(splits) > 1 else [text]

def force_split_at_words(text: str, max_chars: int) -> List[str]:
    """Force split text at word boundaries when no natural split points exist."""
    words = text.split()
    splits = []
    current_segment = ""
    
    for word in words:
        if len(current_segment + " " + word) <= max_chars:
            current_segment += (" " + word) if current_segment else word
        else:
            if current_segment:
                splits.append(current_segment)
                current_segment = word
            else:
                # Single word longer than max_chars, force include
                splits.append(word)
                current_segment = ""
    
    if current_segment:
        splits.append(current_segment)
    
    return splits

def split_at_semantic_breakpoints(text: str, max_chars: int) -> List[str]:
    """Split text at semantic breakpoints when no natural punctuation exists."""
    if not text or not text.strip():
        return []
    
    # Define semantic breakpoint patterns (in priority order)
    breakpoint_patterns = [
        # 1. Breathing pause words (natural speech breaks)
        r'\b(and|but|so|then|well|like|you know|actually|honestly|really)\s+',
        
        # 2. Subordinate clause markers
        r'\s+(that|which|who|where|when|while|since|because|if|unless|although)\s+',
        
        # 3. Topic transition words
        r'\b(funny story|by the way|speaking of|meanwhile|however|therefore|anyway)\s+',
        
        # 4. Time/place transitions  
        r'\b(here|there|now|then|after|before|during|while|when)\s+',
        
        # 5. Explanatory phrases
        r'\b(I mean|in other words|that is|for example|such as)\s+',
        
        # 6. Repeated word gaps (like "food a lot of food")
        r'(\w+)(\s+[^.!?]*?\s+)\1\b'
    ]
    
    for pattern in breakpoint_patterns:
        splits = _split_with_pattern(text, pattern, max_chars)
        if len(splits) > 1:
            return splits
    
    # If no semantic breakpoints found, return original text
    return [text]

def _split_with_pattern(text: str, pattern: str, max_chars: int) -> List[str]:
    """Helper function to split text using a regex pattern."""
    import re
    
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return [text]
    
    splits = []
    current_pos = 0
    
    for match in matches:
        split_pos = match.start()
        
        # Check if splitting here creates reasonable segments
        part_before = text[current_pos:split_pos].strip()
        remaining = text[split_pos:].strip()
        
        if (len(part_before) >= MIN_SEGMENT_LENGTH and 
            len(part_before) <= max_chars and 
            len(remaining) > MIN_SEGMENT_LENGTH):
            splits.append(part_before)
            current_pos = split_pos
    
    # Add remaining text
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            splits.append(remaining)
    
    return splits if len(splits) > 1 else [text]

def create_timed_segments(text_splits: List[str], start_time: float, end_time: float) -> List[Dict]:
    """Create segments with REAL word-level timing - NO MORE GUESSING!
    
    This function should ONLY be called when we don't have word-level timestamps.
    When we do have word timestamps, use create_precise_timed_segments() instead.
    """
    if not text_splits:
        return []
    
    if len(text_splits) == 1:
        return [{
            'text': text_splits[0],
            'start': start_time,
            'end': end_time,
            'word_count': len(text_splits[0].split())
        }]
    
    # FALLBACK ONLY: Use character-based proportional timing
    # This is imprecise but better than nothing when word timestamps unavailable
    logging.warning("Using fallback proportional timing - precision may be reduced")
    
    total_chars = sum(len(split) for split in text_splits)
    segments = []
    current_time = start_time
    duration = end_time - start_time
    
    for i, text_split in enumerate(text_splits):
        if i == len(text_splits) - 1:
            segment_end = end_time
        else:
            ratio = len(text_split) / total_chars
            segment_duration = duration * ratio
            segment_end = current_time + segment_duration
        
        segments.append({
            'text': text_split,
            'start': current_time,
            'end': segment_end,
            'word_count': len(text_split.split())
        })
        
        current_time = segment_end
    
    return segments

def create_precise_timed_segments(text_splits: List[str], original_segment_words: List[Dict]) -> List[Dict]:
    """Create segments with PRECISE word-level timing using sliding window matching.
    
    This is the RIGHT way to do timing - using actual Whisper word timestamps.
    """
    if not text_splits or not original_segment_words:
        logging.error("🚫 TIMESTAMP DEBUG: No text_splits or original_segment_words provided")
        return []
    
    logging.info(f"🎯 TIMESTAMP DEBUG: Processing {len(text_splits)} text splits with {len(original_segment_words)} Whisper words")
    
    # Extract clean words from Whisper data (preserve contractions like "I'm", "don't")
    whisper_clean_words = []
    for i, word_data in enumerate(original_segment_words):
        word = word_data.get('word', '').strip()
        if word:
            # Remove punctuation but keep apostrophes for contractions
            clean_word = re.sub(r"[^\w\s']", '', word).lower().strip()
            if clean_word:
                whisper_clean_words.append({
                    'clean': clean_word,
                    'start': word_data.get('start', 0.0),
                    'end': word_data.get('end', 0.0),
                    'original': word
                })
                logging.debug(f"  🔤 Word {i}: '{word}' → clean: '{clean_word}' ({word_data.get('start', 0.0):.2f}s-{word_data.get('end', 0.0):.2f}s)")
    
    if not whisper_clean_words:
        logging.error("🚫 TIMESTAMP DEBUG: No clean words extracted from Whisper data")
        return []
    
    logging.info(f"📝 TIMESTAMP DEBUG: Extracted {len(whisper_clean_words)} clean Whisper words:")
    whisper_word_list = [w['clean'] for w in whisper_clean_words]
    logging.info(f"   Whisper words: {' '.join(whisper_word_list)}")
    
    segments = []
    used_word_indices = set()
    
    for split_idx, text_split in enumerate(text_splits):
        logging.info(f"\n🔍 TIMESTAMP DEBUG: Processing split {split_idx + 1}/{len(text_splits)}: '{text_split}'")
        
        # Extract clean words from text split (same cleaning as Whisper words)
        split_words = []
        for word in text_split.split():
            clean_word = re.sub(r"[^\w\s']", '', word).lower().strip()
            if clean_word:
                split_words.append(clean_word)
        
        if not split_words:
            logging.warning(f"⚠️  TIMESTAMP DEBUG: No clean words in split {split_idx + 1}")
            continue
        
        logging.info(f"   🎯 Target words: {' '.join(split_words)} (length: {len(split_words)})")
        logging.info(f"   🚫 Used indices: {sorted(used_word_indices)}")
        
        # Find best matching position using sliding window
        best_match = find_best_word_sequence_match(
            split_words, whisper_clean_words, used_word_indices
        )
        
        if best_match:
            start_idx, end_idx, confidence = best_match
            
            # Mark words as used
            used_word_indices.update(range(start_idx, end_idx + 1))
            
            # Get precise timestamps from matched words
            segment_start = whisper_clean_words[start_idx]['start']
            segment_end = whisper_clean_words[end_idx]['end']
            
            matched_whisper_words = [whisper_clean_words[i]['clean'] for i in range(start_idx, end_idx + 1)]
            
            segments.append({
                'text': text_split,
                'start': segment_start,
                'end': segment_end,
                'word_count': len(split_words),
                'timing_confidence': confidence
            })
            
            logging.info(f"✅ DEBUG: SUCCESSFUL MATCH! spaCy sentence -> WhisperX words {start_idx}-{end_idx}")
            logging.info(f"✅ DEBUG: Matched sentence: '{text_split[:60]}{'...' if len(text_split) > 60 else ''}'")
            logging.info(f"✅ DEBUG: Matched timing: {segment_start:.3f}s-{segment_end:.3f}s (confidence: {confidence:.2f})")
            logging.info(f"✅ DEBUG: WhisperX words matched: {matched_whisper_words}")
            logging.info(f"      🎯 Target:  {' '.join(split_words)}")
            logging.info(f"      🎤 Whisper: {' '.join(matched_whisper_words)}")
            
        else:
            logging.error(f"   ❌ NO MATCH FOUND for split {split_idx + 1}: '{text_split[:50]}...'")
            logging.error(f"      🎯 Target words: {' '.join(split_words)}")
            logging.error(f"      🎤 Available Whisper words: {' '.join([w['clean'] for i, w in enumerate(whisper_clean_words) if i not in used_word_indices])}")
            
            # Use fallback timing based on previous segment
            if segments:
                prev_end = segments[-1]['end']
                fallback_duration = len(text_split) * 0.1  # ~10 chars per second estimate
                fallback_start = prev_end
                fallback_end = prev_end + fallback_duration
                
                segments.append({
                    'text': text_split,
                    'start': fallback_start,
                    'end': fallback_end,
                    'word_count': len(split_words),
                    'timing_confidence': 0.0
                })
                
                logging.warning(f"⏰ DEBUG: FALLBACK TIMING TRIGGERED - prev_end: {prev_end:.3f}s, duration estimate: {fallback_duration:.3f}s")
                logging.warning(f"⏰ DEBUG: Fallback segment: '{text_split[:50]}{'...' if len(text_split) > 50 else ''}'")
                logging.warning(f"⏰ DEBUG: Fallback timing: {fallback_start:.3f}s-{fallback_end:.3f}s")
                logging.warning(f"⏰ DEBUG: Reason: Could not match spaCy sentence to WhisperX words")
            else:
                # Very first segment with no match - use segment boundaries
                fallback_start = whisper_clean_words[0]['start']
                fallback_end = whisper_clean_words[0]['end'] + len(text_split) * 0.1
                
                segments.append({
                    'text': text_split,
                    'start': fallback_start,
                    'end': fallback_end,
                    'word_count': len(split_words),
                    'timing_confidence': 0.0
                })
                
                logging.warning(f"⏰ DEBUG: FIRST SEGMENT FALLBACK - using word[0] start: {whisper_clean_words[0]['start']:.3f}s")
                logging.warning(f"⏰ DEBUG: First segment fallback: '{text_split[:50]}{'...' if len(text_split) > 50 else ''}'")
                logging.warning(f"⏰ DEBUG: First segment timing: {fallback_start:.3f}s-{fallback_end:.3f}s")
                logging.warning(f"⏰ DEBUG: Reason: First segment, could not match to WhisperX words")
    
    logging.info(f"\n📊 TIMESTAMP DEBUG: Final results for {len(segments)} segments:")
    for i, seg in enumerate(segments):
        confidence_emoji = "✅" if seg['timing_confidence'] > 0.8 else "⚠️" if seg['timing_confidence'] > 0.0 else "❌"
        logging.info(f"   {confidence_emoji} Segment {i+1}: '{seg['text'][:50]}...' → {seg['start']:.2f}s-{seg['end']:.2f}s (conf: {seg['timing_confidence']:.2f})")
    
    return segments

def find_best_word_sequence_match(target_words: List[str], whisper_words: List[Dict], 
                                 used_indices: set) -> Optional[Tuple[int, int, float]]:
    """Find the best matching word sequence using sliding window with fuzzy matching.
    
    Returns: (start_index, end_index, confidence) or None if no good match found
    """
    if not target_words or not whisper_words:
        logging.debug("      🚫 SLIDING WINDOW: No target_words or whisper_words")
        return None
    
    target_len = len(target_words)
    best_match = None
    best_score = 0.0
    
    logging.debug(f"      🎯 SLIDING WINDOW: Looking for {target_len} words: {' '.join(target_words)}")
    logging.debug(f"      🚫 SLIDING WINDOW: Used indices: {sorted(used_indices)}")
    
    # Try exact matching first (sliding window)
    exact_matches_found = []
    for start_idx in range(len(whisper_words) - target_len + 1):
        # Skip if this range overlaps with already used words
        if any(idx in used_indices for idx in range(start_idx, start_idx + target_len)):
            logging.debug(f"         ⏭️  Skip window [{start_idx}:{start_idx + target_len}] - overlaps used indices")
            continue
        
        # Check exact match
        window_words = [whisper_words[i]['clean'] for i in range(start_idx, start_idx + target_len)]
        
        if window_words == target_words:
            # Perfect exact match
            logging.debug(f"         ✅ PERFECT MATCH at [{start_idx}:{start_idx + target_len - 1}]!")
            return (start_idx, start_idx + target_len - 1, 1.0)
        
        # Calculate fuzzy match score
        matches = sum(1 for i in range(target_len) 
                     if i < len(window_words) and window_words[i] == target_words[i])
        score = matches / target_len
        
        logging.debug(f"         📊 Window [{start_idx}:{start_idx + target_len - 1}]: {' '.join(window_words)} → score: {score:.2f} ({matches}/{target_len})")
        
        # Accept if >80% match and better than previous
        if score > 0.8 and score > best_score:
            best_score = score
            best_match = (start_idx, start_idx + target_len - 1, score)
            logging.debug(f"         ⭐ New best match: score {score:.2f}")
    
    # If no good exact match, try flexible matching (allowing gaps/insertions)
    if not best_match or best_score < 0.9:
        logging.debug(f"      🔄 SLIDING WINDOW: No good exact match (best: {best_score:.2f}), trying flexible matching...")
        flexible_match = find_flexible_word_match(target_words, whisper_words, used_indices)
        if flexible_match and flexible_match[2] > best_score:
            logging.debug(f"         ✨ Flexible match better: {flexible_match[2]:.2f} > {best_score:.2f}")
            best_match = flexible_match
    
    if best_match:
        logging.debug(f"      ✅ SLIDING WINDOW: Final match [{best_match[0]}:{best_match[1]}] with score {best_match[2]:.2f}")
    else:
        logging.debug(f"      ❌ SLIDING WINDOW: No acceptable match found")
    
    return best_match

def find_flexible_word_match(target_words: List[str], whisper_words: List[Dict], 
                           used_indices: set) -> Optional[Tuple[int, int, float]]:
    """Try flexible matching allowing for small gaps or word order changes."""
    if not target_words:
        return None
    
    # Look for the first and last target words to establish boundaries
    first_word = target_words[0]
    last_word = target_words[-1]
    
    first_positions = []
    last_positions = []
    
    for i, word_data in enumerate(whisper_words):
        if i not in used_indices:
            if word_data['clean'] == first_word:
                first_positions.append(i)
            if word_data['clean'] == last_word:
                last_positions.append(i)
    
    best_match = None
    best_score = 0.0
    
    # Try combinations of first and last word positions
    for first_pos in first_positions:
        for last_pos in last_positions:
            if last_pos <= first_pos:
                continue
            
            # Check if range overlaps with used indices
            if any(idx in used_indices for idx in range(first_pos, last_pos + 1)):
                continue
            
            # Calculate how many target words are found in this range
            range_words = [whisper_words[i]['clean'] for i in range(first_pos, last_pos + 1)]
            found_words = sum(1 for word in target_words if word in range_words)
            score = found_words / len(target_words)
            
            if score > best_score:
                best_score = score
                best_match = (first_pos, last_pos, score)
    
    # Only return if we found at least 60% of the words
    if best_match and best_score >= 0.6:
        return best_match
    
    return None

# The structure_and_split_segments function is imported from clean_segmentation.py
# Old complex implementation has been removed and replaced with clean architecture

def _try_precision_path(transcription_result: Dict, segmentation_strategy: str) -> List[Dict]:
    """Try high-precision path: word timestamps + spaCy sentences + precise mapping."""
    logging.info("Attempting high-precision processing path...")
    
    # Step 1: Extract Whisper segments and all words
    whisper_segments = transcription_result.get("segments", [])
    if not whisper_segments:
        logging.warning("No segments found in transcription result")
        return []
    
    # Extract all words and track segment boundaries
    all_words = []
    segment_boundaries = []
    
    for segment in whisper_segments:
        start_word_idx = len(all_words)
        segment_words = segment.get("words", [])
        all_words.extend(segment_words)
        end_word_idx = len(all_words) - 1
        
        if segment_words:  # Only add boundary if segment has words
            segment_boundaries.append({
                'start_word_idx': start_word_idx,
                'end_word_idx': end_word_idx,
                'original_text': segment.get('text', '').strip(),
                'start_time': segment.get('start', 0.0),
                'end_time': segment.get('end', 0.0)
            })
    
    if not all_words:
        logging.warning("No word-level timestamps found")
        return []
    
    logging.info(f"Extracted {len(all_words)} words from {len(segment_boundaries)} segments")
    
    # Step 2: Apply intelligent punctuation
    try:
        final_text, clean_words_for_mapping, clean_reference_text = apply_intelligent_punctuation(
            all_words, segment_boundaries, segmentation_strategy
        )
        
        if not final_text.strip():
            logging.warning("Final punctuated text is empty")
            return []
    except Exception as e:
        logging.warning(f"Punctuation application failed: {e}")
        return []
    
    # Step 3: spaCy sentence boundary detection
    try:
        doc = nlp(final_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            logging.warning("No sentences detected by spaCy")
            return []
            
        logging.info(f"📝 DEBUG: spaCy detected {len(sentences)} sentences")
        logging.info(f"📝 DEBUG: Full text length: {len(final_text)} characters")
        
        # Log first few sentences for comparison
        for i, sent in enumerate(sentences[:3]):
            logging.info(f"📝 DEBUG: spaCy sentence {i+1}: '{sent[:80]}{'...' if len(sent) > 80 else ''}'")
        
        # Compare to WhisperX segment boundaries
        whisper_segments = transcription_result.get("segments", [])
        logging.info(f"📝 DEBUG: WhisperX produced {len(whisper_segments)} segments vs spaCy {len(sentences)} sentences")
        
        for i, seg in enumerate(whisper_segments[:3]):
            logging.info(f"📝 DEBUG: WhisperX segment {i+1}: '{seg.get('text', '')[:80]}{'...' if len(seg.get('text', '')) > 80 else ''}'")
            logging.info(f"📝 DEBUG: WhisperX segment {i+1} timing: {seg.get('start', 0):.3f}s-{seg.get('end', 0):.3f}s")
    except Exception as e:
        logging.warning(f"spaCy processing failed: {e}")
        return []
    
    # Step 4: Map sentences to word timestamps
    try:
        mapped_segments = map_sentences_to_word_timestamps(
            sentences, clean_words_for_mapping, clean_reference_text
        )
        
        if not mapped_segments:
            logging.warning("Failed to map sentences to word timestamps")
            return []
            
        logging.info(f"Successfully mapped {len(mapped_segments)} segments with precise timestamps")
        return mapped_segments
        
    except Exception as e:
        logging.warning(f"Sentence-to-timestamp mapping failed: {e}")
        return []

def _try_fallback_path(transcription_result: Dict, segmentation_strategy: str) -> List[Dict]:
    """Fallback path: Use Whisper segments directly with basic punctuation."""
    logging.info("Using fallback processing path...")
    
    whisper_segments = transcription_result.get("segments", [])
    if not whisper_segments:
        logging.error("No Whisper segments available for fallback")
        return []
    
    fallback_segments = []
    for segment in whisper_segments:
        text = segment.get('text', '').strip()
        if not text:
            continue
        
        # Apply basic punctuation enhancement
        try:
            enhanced_text = apply_rule_based_punctuation(text)
        except Exception as e:
            logging.warning(f"Rule-based punctuation failed, using original: {e}")
            enhanced_text = text
        
        fallback_segments.append({
            'text': enhanced_text,
            'start': segment.get('start', 0.0),
            'end': segment.get('end', 0.0)
        })
    
    logging.info(f"Fallback path produced {len(fallback_segments)} segments")
    return fallback_segments

def _apply_unified_segment_processing(raw_segments: List[Dict], max_chars: int, transcription_result: Dict) -> List[Dict]:
    """CRITICAL: Unified processing that ALL paths must go through.
    
    This ensures every segment is properly sized and has correct timestamps,
    regardless of which path (precision or fallback) produced the raw segments.
    """
    if not raw_segments:
        logging.error("🚫 UNIFIED PROCESSING: No raw segments provided")
        return []
    
    logging.info(f"🔧 UNIFIED PROCESSING: Starting with {len(raw_segments)} raw segments, max_chars={max_chars}")
    
    # Extract ALL word-level data for precise timing
    all_words = []
    whisper_segments = transcription_result.get("segments", [])
    for segment in whisper_segments:
        segment_words = segment.get("words", [])
        if segment_words:
            all_words.extend(segment_words)
    
    logging.info(f"📊 UNIFIED PROCESSING: Extracted {len(all_words)} word timestamps for precise splitting")
    
    final_segments = []
    
    for seg_idx, segment in enumerate(raw_segments):
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']
        
        logging.info(f"\n🔍 UNIFIED PROCESSING: Segment {seg_idx + 1}/{len(raw_segments)}")
        logging.info(f"   📝 Text ({len(text)} chars): '{text[:80]}{'...' if len(text) > 80 else ''}'")
        logging.info(f"   ⏰ Time: {start_time:.2f}s - {end_time:.2f}s")
        
        # Check if segment needs splitting
        if len(text) > max_chars:
            logging.warning(f"   ⚠️  LONG SEGMENT DETECTED! {len(text)} chars > {max_chars} max")
            logging.info(f"   🔧 Applying intelligent splitting with word-level timing...")
            
            # Apply intelligent splitting WITH precise word-level timing
            split_segments = split_long_segment(segment, max_chars, all_words)
            
            if split_segments:
                logging.info(f"   ✅ Split into {len(split_segments)} segments:")
                for i, split_seg in enumerate(split_segments):
                    logging.info(f"      {i+1}. '{split_seg['text'][:40]}...' ({split_seg['start']:.2f}s-{split_seg['end']:.2f}s)")
                final_segments.extend(split_segments)
            else:
                logging.error(f"   ❌ Splitting failed! Keeping original segment")
                final_segments.append(segment)
            
        else:
            logging.info(f"   ✅ Segment size OK ({len(text)} chars ≤ {max_chars})")
            final_segments.append(segment)
    
    # Ensure all segments have required fields and are properly formatted
    logging.info(f"\n🔧 UNIFIED PROCESSING: Validating {len(final_segments)} final segments...")
    
    for i, segment in enumerate(final_segments):
        # Ensure all required fields exist
        if 'word_count' not in segment:
            segment['word_count'] = len(segment['text'].split())
        
        # Validate timestamps
        if segment['start'] >= segment['end']:
            logging.warning(f"   ⚠️  Invalid timestamps in segment {i+1}: {segment['start']:.2f}s >= {segment['end']:.2f}s")
            # Fix by adding minimal duration
            if i > 0:
                segment['start'] = final_segments[i-1]['end']
            segment['end'] = segment['start'] + 1.0
            logging.info(f"      🔧 Fixed to: {segment['start']:.2f}s - {segment['end']:.2f}s")
        
        # Add timing confidence if not present
        if 'timing_confidence' not in segment:
            segment['timing_confidence'] = 1.0 if all_words else 0.5
    
    logging.info(f"\n📊 UNIFIED PROCESSING COMPLETE:")
    logging.info(f"   📥 Input:  {len(raw_segments)} raw segments")
    logging.info(f"   📤 Output: {len(final_segments)} final segments")
    logging.info(f"   🎯 All segments ≤ {max_chars} chars: {'✅ YES' if all(len(s['text']) <= max_chars for s in final_segments) else '❌ NO'}")
    
    # Show final segment summary
    for i, segment in enumerate(final_segments):
        confidence_emoji = "✅" if segment.get('timing_confidence', 0) > 0.8 else "⚠️" if segment.get('timing_confidence', 0) > 0.0 else "❌"
        logging.info(f"   {confidence_emoji} Final {i+1}: '{segment['text'][:50]}...' ({len(segment['text'])} chars, {segment['start']:.2f}s-{segment['end']:.2f}s)")
    
    return final_segments



# Global BERT model and pipeline cache to avoid repeated loading
_bert_model = None
_bert_tokenizer = None
_bert_pipeline = None

def apply_bert_punctuation(text: str) -> str:
    """Apply BERT-based punctuation restoration with caching."""
    global _bert_model, _bert_tokenizer, _bert_pipeline
    
    if not text or not text.strip():
        return text
        
    try:
        # Load model and create pipeline only once for performance
        if _bert_pipeline is None:
            logging.info("Loading BERT model: felflare/bert-restore-punctuation...")
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            _bert_tokenizer = AutoTokenizer.from_pretrained("felflare/bert-restore-punctuation")
            _bert_model = AutoModelForTokenClassification.from_pretrained("felflare/bert-restore-punctuation")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _bert_model.to(device)
            
            # Cache the pipeline for better performance
            # Use "none" aggregation to get raw LABEL_X format
            _bert_pipeline = pipeline("token-classification", 
                                    model=_bert_model, 
                                    tokenizer=_bert_tokenizer,
                                    aggregation_strategy="none",  # Get raw labels
                                    device=device)
            
            logging.info(f"BERT punctuation model loaded successfully on {device}")
        
        # Apply punctuation using cached pipeline
        result = _bert_pipeline(text)
        
        # Reconstruct text with punctuation
        enhanced_text = reconstruct_text_with_punctuation(text, result)
        return enhanced_text
        
    except Exception as e:
        logging.warning(f"BERT punctuation failed: {e}")
        return apply_rule_based_punctuation(text)


def apply_rule_based_punctuation(text: str) -> str:
    """Apply simple rule-based punctuation (fallback for BERT)."""
    if not text:
        return text
        
    text = text.strip()
    if text:
        # Basic capitalization
        text = text[0].upper() + text[1:]
    
    # Add period at the end if no ending punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

def reconstruct_text_with_punctuation(original_text: str, bert_results: List[Dict]) -> str:
    """Reconstruct text with BERT-predicted punctuation."""
    if not bert_results:
        return apply_rule_based_punctuation(original_text)
    
    try:
        # The BERT model outputs token classifications
        # We need to map tokens back to original text with punctuation
        tokens = original_text.split()
        
        
        # FIXED: Correct label mapping for felflare/bert-restore-punctuation model
        correct_label_mapping = {
            0: "O",           # No punctuation
            1: "B-PERIOD",    # Period (.)
            2: "B-COMMA",     # Comma (,)
            3: "B-QUESTION",  # Question mark (?)
            4: "B-EXCLAMATION", # Exclamation mark (!)
            5: "B-SEMICOLON", # Semicolon (;)
            6: "B-COLON",     # Colon (:)
            7: "B-APOSTROPHE", # Apostrophe (')
            8: "B-QUOTATION", # Quotation mark (")
            9: "B-DASH",      # Dash (-)
            10: "B-ELLIPSIS", # Ellipsis (...)
            11: "B-PARENTHESIS_OPEN", # Opening parenthesis (
            12: "B-PARENTHESIS_CLOSE", # Closing parenthesis )
            13: "B-BRACKET_OPEN", # Opening bracket [
            14: "B-BRACKET_CLOSE", # Closing bracket ]
        }
        
        punctuation_map = {
            'B-PERIOD': '.',
            'B-COMMA': ',', 
            'B-QUESTION': '?',
            'B-EXCLAMATION': '!',
            'B-SEMICOLON': ';',
            'B-COLON': ':',
            'B-APOSTROPHE': "'",
            'B-QUOTATION': '"',
            'B-DASH': '-',
            'B-ELLIPSIS': '...',
            'O': ''  # No punctuation
        }
        
        reconstructed_tokens = []
        
        # Process each BERT result - but handle length mismatch
        max_tokens = min(len(tokens), len(bert_results))
        
        for i in range(max_tokens):
            result = bert_results[i]
            token = tokens[i]
            
            # FIXED: Convert BERT entity to correct label using our mapping
            entity = result.get('entity', 'NONE')
            if entity.startswith('LABEL_'):
                # Extract label number and map to correct punctuation label
                try:
                    label_num = int(entity.split('_')[1])
                    label = correct_label_mapping.get(label_num, 'O')
                except:
                    label = 'O'
            else:
                label = entity
            
            # Add the token
            reconstructed_tokens.append(token)
            
            # Add punctuation if predicted
            if label in punctuation_map and punctuation_map[label]:
                reconstructed_tokens.append(punctuation_map[label])
        
        # ⚠️ CRITICAL FIX: Add remaining tokens if BERT results are fewer
        if len(bert_results) < len(tokens):
            for i in range(len(bert_results), len(tokens)):
                reconstructed_tokens.append(tokens[i])
        
        # Join and clean up
        reconstructed = ' '.join(reconstructed_tokens)
        
        # Clean up spacing around punctuation
        reconstructed = re.sub(r'\s+([,.!?;:])', r'\1', reconstructed)
        
        # Ensure proper capitalization
        reconstructed = reconstructed.strip()
        if reconstructed:
            reconstructed = reconstructed[0].upper() + reconstructed[1:]
        
        
        return reconstructed
        
    except Exception as e:
        logging.warning(f"BERT text reconstruction failed: {e}")
        return apply_rule_based_punctuation(original_text)



# --- File Saving Utilities ---
def save_subtitles_to_srt(subs: List[srt.Subtitle], path: str):
    """Save subtitles in SRT format with validation."""
    if not subs:
        logging.warning(f"No subtitles to save to {path}")
        return
    
    # Quick validation before saving
    valid_subs = []
    for sub in subs:
        start_sec = sub.start.total_seconds()
        end_sec = sub.end.total_seconds()
        if start_sec < end_sec and sub.content.strip():
            valid_subs.append(sub)
        else:
            logging.warning(f"Skipping invalid subtitle: {start_sec:.3f}s-{end_sec:.3f}s '{sub.content[:20]}...'")
    
    if not valid_subs:
        logging.error(f"No valid subtitles to save to {path}")
        return
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(valid_subs))
    
    # Log save confirmation
    max_time = max(sub.end.total_seconds() for sub in valid_subs)
    logging.info(f"✓ Saved {len(valid_subs)} subtitles to {Path(path).name} (duration: {max_time:.1f}s)")

def save_subtitles_to_txt(subs: List[srt.Subtitle], path: str):
    """Save subtitles in TXT format."""
    with open(path, "w", encoding="utf-8") as f:
        for s in subs:
            f.write(f"[{s.start} --> {s.end}]\n{s.content}\n\n")

def save_subtitles_to_vtt(subs: List[srt.Subtitle], path: str):
    """Save subtitles in VTT format."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for sub in subs:
            start = str(sub.start).replace(',', '.')
            end = str(sub.end).replace(',', '.')
            f.write(f"{start} --> {end}\n{sub.content}\n\n")

def save_subtitles_to_ass(subs: List[srt.Subtitle], path: str):
    """Save subtitles in ASS format."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("Title: Generated Subtitles\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for sub in subs:
            start_sec = sub.start.total_seconds()
            end_sec = sub.end.total_seconds()
            start_time = f"{int(start_sec//3600)}:{int((start_sec%3600)//60):02d}:{start_sec%60:05.2f}"
            end_time = f"{int(end_sec//3600)}:{int((end_sec%3600)//60):02d}:{end_sec%60:05.2f}"
            content = sub.content.replace('\n', '\\N')
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{content}\n")

def save_subtitles_to_json(subs: List[srt.Subtitle], path: str):
    """Save subtitles in JSON format."""
    data = []
    for s in subs:
        data.append({
            "index": s.index,
            "start": str(s.start),
            "end": str(s.end),
            "content": s.content
        })
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- Whisper Anomaly Detection and Retry ---
def check_and_retry_transcription(result, model, audio_path, args, chunk_name, audio_segment):
    """Check transcription result for anomalies and retry if needed."""
    
    if not result or not isinstance(result, dict) or 'segments' not in result:
        return result
    
    segments = result['segments']
    if not segments:
        return result
    
    # Detect anomalies
    anomalies = detect_transcription_anomalies(segments, chunk_name)
    
    if not anomalies:
        logging.info(f"✅ {chunk_name}: Transcription looks good, no anomalies detected")
        return result
    
    logging.warning(f"⚠️ {chunk_name}: Detected transcription anomalies: {', '.join(anomalies)}")
    
    # Retry transcription with different parameters
    retry_count = 0
    max_retries = 2
    
    while retry_count < max_retries:
        retry_count += 1
        logging.info(f"🔄 {chunk_name}: Retry attempt {retry_count}/{max_retries}")
        
        # Try different Whisper parameters
        retry_params = get_retry_parameters(retry_count, args)
        
        try:
            retry_result = model.transcribe(
                audio_path,
                language=args.source_language,
                word_timestamps=args.word_timestamps,
                fp16=torch.cuda.is_available(),
                verbose=False,  # Less verbose for retries
                **retry_params
            )
            
            if retry_result and 'segments' in retry_result:
                retry_anomalies = detect_transcription_anomalies(retry_result['segments'], chunk_name)
                
                if len(retry_anomalies) < len(anomalies):
                    logging.info(f"✅ {chunk_name}: Retry {retry_count} improved quality (anomalies: {len(anomalies)} -> {len(retry_anomalies)})")
                    return retry_result
                else:
                    logging.info(f"⚠️ {chunk_name}: Retry {retry_count} didn't improve quality")
            
        except Exception as e:
            logging.error(f"❌ {chunk_name}: Retry {retry_count} failed: {e}")
    
    logging.warning(f"⚠️ {chunk_name}: All retries exhausted, using original result with anomalies")
    return result


def detect_transcription_anomalies(segments, chunk_name):
    """Detect various transcription anomalies."""
    anomalies = []
    
    if not segments:
        return anomalies
    
    # 1. Check for repeated segments
    repeated_count = 0
    for i in range(1, len(segments)):
        current_text = segments[i].get('text', '').strip()
        prev_text = segments[i-1].get('text', '').strip()
        
        if current_text and prev_text and current_text == prev_text:
            repeated_count += 1
    
    if repeated_count > 1:  # More than 1 repeat is suspicious
        anomalies.append(f"repeated_segments({repeated_count})")
    
    # 2. Check for abnormal segment durations
    long_segments = 0
    zero_duration_segments = 0
    
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        duration = end - start
        
        if duration > 15:  # Segment longer than 15 seconds
            long_segments += 1
        elif duration <= 0:  # Zero or negative duration
            zero_duration_segments += 1
    
    if long_segments > 0:
        anomalies.append(f"long_segments({long_segments})")
    if zero_duration_segments > 0:
        anomalies.append(f"zero_duration({zero_duration_segments})")
    
    # 3. Check for fragmented content (too many short segments)
    short_segments = 0
    single_char_segments = 0
    
    for seg in segments:
        text = seg.get('text', '').strip()
        if len(text) <= 3:  # Very short text
            short_segments += 1
            if len(text) == 1:
                single_char_segments += 1
    
    # If more than 30% of segments are very short, it's suspicious
    if len(segments) > 5 and short_segments / len(segments) > 0.3:
        anomalies.append(f"fragmented({short_segments}/{len(segments)})")
    
    if single_char_segments > 2:
        anomalies.append(f"single_chars({single_char_segments})")
    
    # 4. Check for timestamp overlaps
    overlaps = 0
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get('end', 0)
        curr_start = segments[i].get('start', 0)
        
        if curr_start < prev_end:  # Overlap detected
            overlaps += 1
    
    if overlaps > 0:
        anomalies.append(f"overlaps({overlaps})")
    
    return anomalies


def get_retry_parameters(retry_count, original_args):
    """Get different parameters for retry attempts."""
    
    if retry_count == 1:
        # First retry: Increase temperature and beam size for more diversity
        return {
            'temperature': 0.2,  # Default is 0, higher = more creative
            'beam_size': 3,      # Default is None, higher = more thorough
            'no_speech_threshold': 0.8,  # More lenient than original 1.0
            'logprob_threshold': -1.5     # More lenient than original -2.0
        }
    elif retry_count == 2:
        # Second retry: Different strategy - more conservative
        return {
            'temperature': 0.0,   # More deterministic
            'beam_size': 1,       # Faster, more focused
            'no_speech_threshold': 0.6,   # Even more lenient
            'logprob_threshold': -1.0,     # Most lenient
            'compression_ratio_threshold': 1.5  # Allow more repetition
        }
    else:
        return {}


# --- Progress Management ---
def save_progress(output_dir: Path, processed_chunks: List[int], all_subs: List[srt.Subtitle]):
    """Save processing progress."""
    progress_file = output_dir / ".progress.json"
    progress_data = {
        "processed_chunks": processed_chunks,
        "total_subtitles": len(all_subs),
        "last_updated": datetime.datetime.now().isoformat()
    }
    
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Failed to save progress: {e}")

def load_progress(output_dir: Path) -> Dict:
    """Load processing progress."""
    progress_file = output_dir / ".progress.json"
    if not progress_file.exists():
        return {"processed_chunks": [], "total_subtitles": 0}
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load progress: {e}")
        return {"processed_chunks": [], "total_subtitles": 0}

# --- Core Processing Logic ---
def process_single_chunk(chunk_path: Path, model: Any, args: argparse.Namespace, glossary: dict, timing_info: Dict) -> tuple:
    """Process a single audio chunk and return subtitles with actual duration."""
    chunk_name = chunk_path.name
    logging.info(f"🔍 {chunk_name} DEBUG: === Starting process_single_chunk ===")
    
    preprocessed_files_to_keep = []
    
    # 1. Audio Preprocessing
    logging.info(f"🔍 {chunk_name} DEBUG: Loading audio from {chunk_path}")
    try:
        audio = AudioSegment.from_file(chunk_path)
        logging.info(f"🔍 {chunk_name} DEBUG: Audio loaded successfully, duration = {len(audio)/1000:.2f}s")
    except Exception as e:
        logging.error(f"🔥 {chunk_name} FAILED: Audio loading failed: {e}")
        return [], 0.0, []
    
    audio_to_transcribe = str(chunk_path)
    
    if not args.no_audio_preprocessing:
        logging.info("Applying audio preprocessing...")
        
        if not args.no_normalize_audio:
            audio = normalize_audio_volume(audio, args.target_dbfs)
        
        if not args.no_denoise:
            use_ai = getattr(args, 'use_ai_vocal_separation', False)
            audio = denoise_audio(audio, args.noise_reduction_strength, use_ai_separation=use_ai)
        
        if not args.no_speaker_detection:
            changes = detect_speaker_changes(audio, args.min_speaker_duration)
            if changes:
                logging.info(f"Detected {len(changes)} speaker changes.")
        
        # Save preprocessed audio
        preprocessed_path = chunk_path.parent.parent / "chunks_preprocessed" / f"{chunk_path.stem}_preprocessed.wav"
        preprocessed_path.parent.mkdir(exist_ok=True)
        audio.export(preprocessed_path, format="wav")
        audio_to_transcribe = str(preprocessed_path)
        preprocessed_files_to_keep.append(preprocessed_path)
    
    # 2. Transcription with Caching
    audio_hash = get_audio_hash(Path(audio_to_transcribe))
    args_hash = hashlib.md5(f"{args.source_language}_{args.max_subtitle_chars}_{args.word_timestamps}".encode()).hexdigest()[:8]
    cache_path = get_cache_path(audio_hash, args.model, args_hash)
    
    result = None
    if is_cache_valid(cache_path):
        logging.info(f"Loading transcription from cache: {cache_path.name}")
        result = load_from_cache(cache_path)
        if result is None:
            logging.warning("Failed to load from cache, will transcribe fresh")
    
    if result is None:
        logging.info(f"🔍 {chunk_name} DEBUG: Starting WhisperX transcription...")
        logging.info(f"🔍 {chunk_name} DEBUG: Audio file = {audio_to_transcribe}")
        logging.info(f"🔍 {chunk_name} DEBUG: Language = {args.source_language}")
        logging.info(f"🔍 {chunk_name} DEBUG: Word timestamps = {args.word_timestamps}")
        
        try:
            try:
                from src.asr.whisper_backend import transcribe_and_align_with_retries
            except ImportError:
                from asr.whisper_backend import transcribe_and_align_with_retries
            result = transcribe_and_align_with_retries(
                audio_to_transcribe,
                language=args.source_language or "en",
                model_size=args.model,
                max_retries=2,
            )
            
            logging.info(f"🔍 {chunk_name} DEBUG: WhisperX transcription completed")
            logging.info(f"🔍 {chunk_name} DEBUG: Result type = {type(result)}")
            logging.info(f"🔍 {chunk_name} DEBUG: Result is None = {result is None}")
            
            if result:
                logging.info(f"🔍 {chunk_name} DEBUG: Result keys = {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                if isinstance(result, dict) and 'segments' in result:
                    logging.info(f"🔍 {chunk_name} DEBUG: Found {len(result['segments'])} segments")
                    for i, seg in enumerate(result['segments'][:3]):  # 只log前3个segment
                        logging.info(f"🔍 {chunk_name} DEBUG: Segment {i}: '{seg.get('text', 'NO_TEXT')[:50]}...'")
                        if 'words' in seg:
                            logging.info(f"🔍 {chunk_name} DEBUG: Segment {i} has {len(seg['words'])} words")
                        else:
                            logging.info(f"🔍 {chunk_name} DEBUG: Segment {i} has NO WORDS")
                # Already retried in WhisperX backend; cache result now
                save_to_cache(cache_path, result)
            else:
                # This is the critical failure point for chunk_1
                logging.error(f"🔥 {chunk_name} FAILED: WhisperX returned None result (silence or no speech detection)")
                logging.error(f"🔥 {chunk_name} FAILED: FORCING subtitle generation anyway to prevent missing chunk")
                
                # Create a minimal fallback subtitle to avoid losing this chunk entirely
                duration_sec = len(audio) / 1000.0
                chunk_start_time_sec = timing_info.get('start_time_ms', 0) / 1000.0
                fallback_subtitle = srt.Subtitle(
                    index=1,
                    start=datetime.timedelta(seconds=chunk_start_time_sec),
                    end=datetime.timedelta(seconds=chunk_start_time_sec + duration_sec),
                    content="[No speech detected in this segment]"
                )
                logging.info(f"🔥 {chunk_name} FAILED: Created fallback subtitle with proper timestamps to prevent chunk loss")
                return [fallback_subtitle], duration_sec, preprocessed_files_to_keep

        except Exception as e:
            import traceback
            logging.error(f"🔥 {chunk_name} FAILED: Transcription exception: {e}")
            logging.error(f"🔥 {chunk_name} FAILED: Transcription traceback: {traceback.format_exc()}")
            logging.error(f"🔥 {chunk_name} FAILED: FORCING fallback subtitle generation to prevent chunk loss")
            
            # Create a fallback subtitle even if transcription completely fails
            duration_sec = len(audio) / 1000.0
            chunk_start_time_sec = timing_info.get('start_time_ms', 0) / 1000.0
            fallback_subtitle = srt.Subtitle(
                index=1,
                start=datetime.timedelta(seconds=chunk_start_time_sec),
                end=datetime.timedelta(seconds=chunk_start_time_sec + duration_sec),
                content=f"[Transcription failed: {str(e)[:100]}]"
            )
            return [fallback_subtitle], duration_sec, preprocessed_files_to_keep
    
    # Add debug log
    logging.info(f"Whisper result type: {type(result)}")
    if result:
        logging.info(f"Whisper result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict) and 'segments' in result:
            logging.info(f"Number of segments: {len(result['segments'])}")
            if result['segments']:
                logging.info(f"First segment: {result['segments'][0]}")
        else:
            logging.info("No segments found in result")
    
    if not result or 'segments' not in result or not result['segments']:
        logging.warning("No transcription result or empty segments found. Skipping subtitle generation for this chunk.")
        return [], len(audio) / 1000.0, preprocessed_files_to_keep
    
    # 3. Structure and format subtitles
    logging.info(f"🔍 {chunk_name} DEBUG: Starting segmentation with strategy '{args.segmentation_strategy}'")
    logging.info(f"🔍 {chunk_name} DEBUG: Max chars = {args.max_subtitle_chars}")
    
    try:
        # New unified pipeline: segmentation (spaCy/Whisper) + long-split + strict alignment
        try:
            from src.processing.pipeline import generate_segments_with_alignment
        except ImportError:
            from pipeline import generate_segments_with_alignment
        structured_segments = generate_segments_with_alignment(
            result, args.max_subtitle_chars, args.segmentation_strategy
        )
        if structured_segments:
            logging.info(f"🔍 {chunk_name} DEBUG: Alignment pipeline produced {len(structured_segments)} segments")
        else:
            # Fallback: use Whisper segments directly, then unified post-processing to enforce length
            logging.warning(f"⚠️ {chunk_name} Alignment pipeline returned no segments; falling back to Whisper segments + unified processing")
            whisper_segments = result.get('segments', []) or []
            raw_segments = []
            for seg in whisper_segments:
                text = (seg.get('text') or '').strip()
                if not text:
                    continue
                try:
                    enhanced = apply_rule_based_punctuation(text)
                except Exception:
                    enhanced = text
                raw_segments.append({
                    'text': enhanced,
                    'start': float(seg.get('start', 0.0)),
                    'end': float(seg.get('end', 0.0)),
                })
            structured_segments = _apply_unified_segment_processing(raw_segments, args.max_subtitle_chars, result)
            logging.info(f"🔍 {chunk_name} DEBUG: Fallback produced {len(structured_segments)} segments")
    except Exception as e:
        import traceback
        logging.error(f"🔥 {chunk_name} FAILED: New alignment pipeline failed: {e}")
        logging.error(f"🔥 {chunk_name} FAILED: Traceback: {traceback.format_exc()}")
        logging.error(f"🔥 {chunk_name} FAILED: FORCING fallback subtitle generation to prevent chunk loss")
        
        # Even if pipeline fails, ensure the chunk produces at least one subtitle to avoid loss
        duration_sec = len(audio) / 1000.0
        chunk_start_time_sec = timing_info.get('start_time_ms', 0) / 1000.0
        fallback_subtitle = srt.Subtitle(
            index=1,
            start=datetime.timedelta(seconds=chunk_start_time_sec),
            end=datetime.timedelta(seconds=chunk_start_time_sec + duration_sec),
            content=f"[Segmentation failed: {str(e)[:100]}]"
        )
        return [fallback_subtitle], duration_sec, preprocessed_files_to_keep
    
    # Extract chunk start time for proper timestamp offset with enhanced validation
    chunk_start_time_ms = timing_info.get('start_time_ms', 0)
    chunk_start_time_sec = chunk_start_time_ms / 1000.0
    chunk_duration_ms = timing_info.get('duration_ms', 0)
    chunk_duration_sec = chunk_duration_ms / 1000.0
    chunk_end_time_sec = chunk_start_time_sec + chunk_duration_sec
    
    logging.info(f"📊 {chunk_name} TIMING VALIDATION:")
    logging.info(f"   Chunk index: {timing_info.get('chunk_index', 'unknown')}")
    logging.info(f"   Chunk start: {chunk_start_time_sec:.3f}s ({chunk_start_time_ms}ms)")
    logging.info(f"   Chunk duration: {chunk_duration_sec:.3f}s ({chunk_duration_ms}ms)")
    logging.info(f"   Chunk expected end: {chunk_end_time_sec:.3f}s")
    logging.info(f"   Audio file duration: {len(audio)/1000:.3f}s")
    
    subtitles = []
    for i, seg in enumerate(structured_segments):
        logging.info(f"🔍 {chunk_name} DEBUG: Processing structured segment {i}")
        logging.info(f"🔍 {chunk_name} DEBUG: Segment keys = {list(seg.keys()) if isinstance(seg, dict) else 'Not a dict'}")
        
        if not isinstance(seg, dict) or 'text' not in seg:
            logging.error(f"🔥 {chunk_name} FAILED: Invalid segment structure: {seg}")
            continue
            
        original = seg['text'].strip()
        logging.info(f"🔍 {chunk_name} DEBUG: Segment text: '{original[:50]}...' (length: {len(original)})")
        if not original:
            logging.info(f"🔍 {chunk_name} DEBUG: Skipping empty segment text")
            continue
        
        # Apply glossary
        processed = apply_glossary(original, glossary["exact_replace"])
        
        # Translate if needed (independent of output format)
        translated = ""
        if (args.translate_during_detection and 
            args.target_language != 'none'):  # Only depends on whether translation is enabled
            translated = translate_text(
                processed, 
                args.target_language, 
                glossary["pre_translate"], 
                args.source_language or 'auto'
            )
        
        # Finalize text
        final_processed = finalize_subtitle_text(processed)
        final_translated = finalize_subtitle_text(translated)
        
        # Create content based on output format (independent of what was generated)
        if args.output_format == 'bilingual':
            if final_translated:
                content = f"{final_processed}\n{final_translated}"
            else:
                # No translation generated, show source only
                content = final_processed
        elif args.output_format == 'target':
            if final_translated:
                content = final_translated
            else:
                # No translation available, fallback to source
                content = final_processed
        else:  # source
            content = final_processed
        
        # Apply chunk time offset to get absolute timestamps with validation
        relative_start = seg.get('start', 0.0)
        relative_end = seg.get('end', 0.0)
        absolute_start = chunk_start_time_sec + relative_start
        absolute_end = chunk_start_time_sec + relative_end
        
        logging.info(f"🔍 {chunk_name} Segment {i} RAW DATA: {seg}")
        
        # Validate segment timing
        if relative_end <= relative_start:
            logging.warning(f"⚠️ {chunk_name} Invalid segment timing: end ({relative_end:.3f}s) <= start ({relative_start:.3f}s)")
            continue
        if relative_end > chunk_duration_sec + 10.0:  # Allow 10s tolerance for debugging
            logging.error(f"🚨 {chunk_name} CRITICAL: Segment extends WAY beyond chunk duration!")
            logging.error(f"   Relative end: {relative_end:.3f}s")
            logging.error(f"   Chunk duration: {chunk_duration_sec:.3f}s")
            logging.error(f"   Difference: {relative_end - chunk_duration_sec:.3f}s")
            logging.error(f"   This is likely the cause of the '2 hour problem'!")
            
        if absolute_end > chunk_end_time_sec + 10.0:  # Check absolute timing too
            logging.error(f"🚨 {chunk_name} CRITICAL: Absolute timestamp way beyond expected!")
            logging.error(f"   Absolute end: {absolute_end:.3f}s ({absolute_end/60:.1f}min)")
            logging.error(f"   Expected chunk end: {chunk_end_time_sec:.3f}s ({chunk_end_time_sec/60:.1f}min)")
            
        logging.info(f"🕰️ {chunk_name} Segment {i}: relative={relative_start:.3f}-{relative_end:.3f}s, absolute={absolute_start:.3f}-{absolute_end:.3f}s, duration={relative_end-relative_start:.3f}s")
        
        subtitle = srt.Subtitle(
            index=0,  # Will be re-indexed later
            start=datetime.timedelta(seconds=absolute_start),
            end=datetime.timedelta(seconds=absolute_end),
            content=content
        )
        subtitles.append(subtitle)
    
    # --- Enhanced validation step for subtitles ---
    validated_subtitles = []
    max_timestamp = 0.0
    
    for sub in subtitles:
        start_sec = sub.start.total_seconds()
        end_sec = sub.end.total_seconds()
        
        if start_sec < end_sec:
            validated_subtitles.append(sub)
            max_timestamp = max(max_timestamp, end_sec)
        else:
            logging.warning(
                f"Invalid subtitle timing detected and skipped: "
                f"start={start_sec:.3f}s, end={end_sec:.3f}s, content='{sub.content[:20]}...'"
            )
    
    actual_duration_sec = len(audio) / 1000.0
    
    # Use the more reliable timing info from chunk_timing.json
    # Instead of actual audio duration which may have been altered by preprocessing
    expected_chunk_end = chunk_end_time_sec
    expected_max_reasonable = chunk_start_time_sec + actual_duration_sec
    
    logging.info(f"📊 {chunk_name} FINAL TIMING SUMMARY:")
    logging.info(f"   Generated subtitles: {len(validated_subtitles)}")
    logging.info(f"   Max subtitle timestamp: {max_timestamp:.3f}s")
    logging.info(f"   Expected chunk end (timing_info): {expected_chunk_end:.3f}s")
    logging.info(f"   Expected max (audio length): {expected_max_reasonable:.3f}s")
    logging.info(f"   Chunk start: {chunk_start_time_sec:.3f}s")
    logging.info(f"   Actual audio duration: {actual_duration_sec:.3f}s")
    logging.info(f"   Timing info duration: {chunk_duration_sec:.3f}s")
    
    # More lenient validation - use the larger of the two expected values plus tolerance
    max_reasonable_timestamp = max(expected_chunk_end, expected_max_reasonable) + 10.0  # 10s tolerance
    
    if max_timestamp > max_reasonable_timestamp:
        logging.warning(f"⚠️ Subtitle timestamps significantly beyond reasonable range: {max_timestamp:.3f}s > {max_reasonable_timestamp:.3f}s")
        logging.warning(f"   This may indicate a serious timestamp calculation error")
    elif max_timestamp > expected_chunk_end + 5.0:  # 5s tolerance for normal variance
        logging.info(f"ℹ️ Subtitle timestamps extend slightly beyond chunk boundary: {max_timestamp:.3f}s > {expected_chunk_end:.3f}s (likely due to preprocessing)")
    
    return validated_subtitles, actual_duration_sec, preprocessed_files_to_keep

def process_chunk_with_offset(task_info: dict) -> dict:
    """Wrapper for parallel execution that loads model inside thread."""
    chunk_index = task_info['index']
    chunk_path = task_info['path']
    args = task_info['args']
    glossary = task_info['glossary']
    timing_info = task_info['timing_info']
    
    # DEBUG: 添加详细的chunk处理跟踪
    logging.info(f"🎯 CHUNK {chunk_index} DEBUG: Starting processing")
    logging.info(f"🎯 CHUNK {chunk_index} DEBUG: Audio path = {chunk_path}")
    logging.info(f"🎯 CHUNK {chunk_index} DEBUG: Audio exists = {chunk_path.exists()}")
    if chunk_path.exists():
        audio_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        logging.info(f"🎯 CHUNK {chunk_index} DEBUG: Audio size = {audio_size_mb:.2f} MB")
    
    try:
        # Use WhisperX backend inside the thread (no preloaded Whisper model)
        logging.info(f"🎯 CHUNK {chunk_index} DEBUG: Starting process_single_chunk with WhisperX backend...")
        subs, actual_duration, preprocessed_files = process_single_chunk(
            chunk_path, None, args, glossary, timing_info
        )
        logging.info(f"🎯 CHUNK {chunk_index} DEBUG: process_single_chunk completed")
        logging.info(f"🎯 CHUNK {chunk_index} DEBUG: Generated {len(subs)} subtitles")
        
        return {
            "index": chunk_index,
            "subs": subs,
            "actual_duration": actual_duration,
            "preprocessed_files": preprocessed_files,
            "status": "ok"
        }
    except Exception as e:
        import traceback
        logging.error(f"🔥 CHUNK {chunk_index} FAILED: {e}")
        logging.error(f"🔥 CHUNK {chunk_index} TRACEBACK: {traceback.format_exc()}")
        return {
            "index": chunk_index,
            "subs": [],
            "actual_duration": 0.0,
            "preprocessed_files": [],
            "status": "error"
        }

# --- Main Workflow ---
def run_main_workflow(args: argparse.Namespace):
    """Main processing workflow."""
    # Setup paths
    paths = get_project_paths(args.filename)
    if not paths["input_file"].exists():
        logging.error(f"Input file not found: {args.filename}")
        sys.exit(1)
    
    # Extract audio if needed
    if not paths["extracted_audio"].exists():
        logging.info("Extracting audio from input file...")
        
        # First, probe the input file for detailed info
        probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(paths["input_file"])]
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            import json
            probe_data = json.loads(probe_result.stdout)
            
            # Log original file info
            if 'format' in probe_data:
                duration = float(probe_data['format'].get('duration', 0))
                logging.info(f"Original file duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
            
            # Find audio streams
            audio_streams = [s for s in probe_data.get('streams', []) if s.get('codec_type') == 'audio']
            for i, stream in enumerate(audio_streams):
                logging.info(f"Audio stream {i}: {stream.get('codec_name')}, {stream.get('sample_rate')}Hz, {stream.get('channels')} channels")
                
        except Exception as e:
            logging.warning(f"Could not probe input file: {e}")
        
        cmd = [
            'ffmpeg', '-i', str(paths["input_file"]),
            '-vn', '-acodec', 'pcm_s16le', 
            '-y', str(paths["extracted_audio"])
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info("Audio extraction completed.")
            
            # Verify extracted audio
            extracted_audio = AudioSegment.from_file(paths["extracted_audio"])
            extracted_duration = len(extracted_audio) / 1000.0
            logging.info(f"Extracted audio duration: {extracted_duration:.2f} seconds ({extracted_duration/60:.1f} minutes)")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Audio extraction failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logging.error("ffmpeg not found. Please install ffmpeg and add it to PATH.")
            sys.exit(1)
    
    # Load glossary
    glossary = load_glossary(paths["output_dir"])
    
    # Plan and split audio
    timing_info = plan_and_split_audio(paths["extracted_audio"], paths, args)
    
    # Create chunk start times map using EXACT timing data from audio splitting
    chunk_start_times = {}
    for timing in timing_info:
        chunk_index = timing['chunk_index']
        # Use the precise start time calculated during audio splitting
        chunk_start_times[chunk_index] = timing['start_time_ms'] / 1000.0
    
    # Prepare tasks for parallel processing
    tasks = []
    for timing in timing_info:
        chunk_index = timing['chunk_index']
        chunk_path = paths["raw_chunks_dir"] / f"chunk_{chunk_index}.wav"
        
        if chunk_path.exists():
            tasks.append({
                "index": chunk_index,
                "path": chunk_path,
                "args": args,
                "glossary": glossary,
                "timing_info": timing
            })
    
    if not tasks:
        logging.error("No audio chunks found to process")
        sys.exit(1)
    
    # Process chunks with parallel processing and error handling
    all_results = []
    initial_max_workers = 1 if args.no_parallel_processing else min(args.max_workers, len(tasks))
    max_workers = initial_max_workers
    
    # Try parallel processing first, fall back to single worker if needed
    while max_workers >= 1:
        try:
            logging.info(f"Starting processing with {max_workers} worker(s)...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 创建future到task的映射来正确跟踪哪个chunk失败了
                future_to_task = {executor.submit(process_chunk_with_offset, task): task for task in tasks}
                
                completed_results = []
                failed_tasks = []
                
                for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing chunks"):
                    task = future_to_task[future]
                    chunk_index = task['index']
                    try:
                        result = future.result()
                        completed_results.append(result)
                        logging.info(f"✅ CHUNK {chunk_index} completed successfully")
                    except Exception as e:
                        logging.error(f"🔥 CHUNK {chunk_index} failed with exception: {e}")
                        failed_tasks.append(task)
                
                all_results = completed_results
                
                # If we have failures and using multiple workers, try single worker for failed tasks
                if failed_tasks and max_workers > 1:
                    failed_chunk_indices = [task['index'] for task in failed_tasks]
                    logging.warning(f"Chunks {failed_chunk_indices} failed with {max_workers} workers. Retrying with single worker...")
                    
                    with ThreadPoolExecutor(max_workers=1) as single_executor:
                        retry_future_to_task = {single_executor.submit(process_chunk_with_offset, task): task for task in failed_tasks}
                        
                        for future in tqdm(as_completed(retry_future_to_task), total=len(failed_tasks), desc="Retrying failed chunks"):
                            task = retry_future_to_task[future]
                            chunk_index = task['index']
                            try:
                                result = future.result()
                                all_results.append(result)
                                logging.info(f"✅ CHUNK {chunk_index} retry succeeded")
                            except Exception as e:
                                logging.error(f"🔥 CHUNK {chunk_index} retry also failed: {e}")
                
                break  # Success, exit retry loop
                
        except Exception as e:
            if max_workers > 1:
                logging.error(f"Parallel processing with {max_workers} workers failed: {e}")
                logging.info("Falling back to single worker processing...")
                max_workers = 1
            else:
                logging.error(f"Even single worker processing failed: {e}")
                break
    
    # Sort results by chunk index for proper sequence
    all_results.sort(key=lambda r: r['index'])
    
    # Log processing summary with enhanced metrics
    successful_chunks = len([r for r in all_results if r['status'] == 'ok'])
    failed_chunks = len([r for r in all_results if r['status'] == 'error'])
    total_chunks = len(tasks)
    
    logging.info(f"📊 CHUNK PROCESSING SUMMARY:")
    logging.info(f"   Successful: {successful_chunks}/{total_chunks} chunks")
    if failed_chunks > 0:
        logging.warning(f"   Failed: {failed_chunks} chunks")
        failed_indices = [r['index'] for r in all_results if r['status'] == 'error']
        logging.warning(f"   Failed chunk indices: {failed_indices}")
    
    # Combine results and apply timing offsets with validation
    all_subs = []
    processed_chunks = []
    all_preprocessed_files = []
    timing_validation_warnings = 0
    
    for result in all_results:
        if result['status'] == 'ok' and result['subs']:
            chunk_index = result['index']
            start_time_sec = chunk_start_times.get(chunk_index, 0.0)
            
            # FIXED: Remove duplicate time offset application
            # Subtitles already have absolute timestamps from process_single_chunk()
            chunk_sub_count = 0
            for sub in result['subs']:
                start_sec = sub.start.total_seconds()
                end_sec = sub.end.total_seconds()
                
                # Validate timing (should already be absolute timestamps)
                if start_sec >= end_sec:
                    logging.warning(f"⚠️ Chunk {chunk_index} invalid timing: {start_sec:.3f}s >= {end_sec:.3f}s")
                    timing_validation_warnings += 1
                    continue
                
                all_subs.append(sub)
                chunk_sub_count += 1
            
            logging.info(f"✓ Chunk {chunk_index}: {chunk_sub_count} subtitles, already absolute timestamps")
            
            processed_chunks.append(chunk_index)
            all_preprocessed_files.extend(result['preprocessed_files'])
            
            # Save individual chunk subtitles
            chunk_output_path = paths["output_dir"] / f"chunk_{chunk_index}.srt"
            save_subtitles_to_srt(result['subs'], str(chunk_output_path))
    
    if not all_subs:
        logging.error("No subtitles were generated")
        sys.exit(1)
    
    # Re-index all subtitles
    for i, sub in enumerate(all_subs):
        sub.index = i + 1
    
    # Save final merged subtitles
    logging.info(f"Saving final subtitles with {len(all_subs)} entries...")
    save_subtitles_to_srt(all_subs, str(paths["final_srt"]))
    
    # Save additional formats if requested
    if args.txt:
        save_subtitles_to_txt(all_subs, str(paths["final_srt"].with_suffix(".txt")))
    if args.vtt:
        save_subtitles_to_vtt(all_subs, str(paths["final_srt"].with_suffix(".vtt")))
    if args.ass:
        save_subtitles_to_ass(all_subs, str(paths["final_srt"].with_suffix(".ass")))
    if args.json:
        save_subtitles_to_json(all_subs, str(paths["final_srt"].with_suffix(".json")))
    
    # Save progress
    if not args.no_save_progress:
        save_progress(paths["output_dir"], processed_chunks, all_subs)
    
    # Handle preprocessed files cleanup
    if args.cleanup_preprocessed:
        for file_path in all_preprocessed_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logging.info(f"Cleaned up preprocessed file: {file_path.name}")
            except Exception as e:
                logging.warning(f"Failed to cleanup {file_path}: {e}")
    else:
        logging.info(f"Kept {len(all_preprocessed_files)} preprocessed files for reprocessing")
    
    # Final comprehensive validation and summary
    if all_subs:
        max_timestamp = max(sub.end.total_seconds() for sub in all_subs)
        total_duration_hours = max_timestamp / 3600.0
        
        # Get original audio duration for validation
        try:
            original_audio = AudioSegment.from_file(paths["extracted_audio"])
            extracted_duration = len(original_audio) / 1000.0
            original_duration_hours = extracted_duration / 3600.0
        except Exception as e:
            logging.warning(f"Could not get original audio duration for validation: {e}")
            extracted_duration = max_timestamp  # Fallback
            original_duration_hours = total_duration_hours
        
        logging.info(f"🏁 FINAL PROCESSING VALIDATION:")
        logging.info(f"   Original audio: {extracted_duration:.1f}s ({extracted_duration/60:.1f}min, {original_duration_hours:.2f}h)")
        logging.info(f"   Subtitle span: {max_timestamp:.1f}s ({max_timestamp/60:.1f}min, {total_duration_hours:.2f}h)")
        logging.info(f"   Total subtitles: {len(all_subs)}")
        logging.info(f"   Processed chunks: {successful_chunks}/{total_chunks}")
        
        duration_diff = abs(max_timestamp - extracted_duration)
        if duration_diff > 60.0:  # More than 1 minute difference
            logging.warning(f"⚠️ DURATION MISMATCH DETECTED!")
            logging.warning(f"   Difference: {duration_diff:.1f}s ({duration_diff/60:.1f}min)")
            logging.warning(f"   This may indicate timing calculation issues!")
        
        # Check for potential timestamp duplication
        timestamps = [sub.start.total_seconds() for sub in all_subs]
        unique_timestamps = set(timestamps)
        if len(timestamps) != len(unique_timestamps):
            duplicates = len(timestamps) - len(unique_timestamps)
            logging.warning(f"⚠️ DUPLICATE TIMESTAMPS: {duplicates} duplicate start times detected")
    
    logging.info(f"✨ Processing completed successfully! ✨")
    logging.info(f"Final subtitles saved to: {paths['final_srt']}")
    logging.info(f"Total subtitles generated: {len(all_subs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subtitles from audio/video files with advanced features.")
    
    # Core arguments
    parser.add_argument("filename", help="Input audio or video file")
    parser.add_argument("--model", default="large-v3", help="WhisperX model size (tiny, base, small, medium, large, large-v2, large-v3)")
    parser.add_argument("--target_language", default="none", help="Target language for translation")
    parser.add_argument("--source_language", default=None, help="Source language code (auto-detect if not set)")
    parser.add_argument("--output_format", default="source", choices=['bilingual', 'source', 'target'], help="Subtitle output format")
    parser.add_argument("--translate_during_detection", action="store_true", help="Enable translation during processing")
    
    # Chunking and processing
    parser.add_argument("--long_audio_threshold", type=int, default=DEFAULT_LONG_AUDIO_THRESHOLD, help="Minutes threshold for chunking audio")
    parser.add_argument("--chunk_duration", type=int, default=DEFAULT_CHUNK_DURATION, help="Target chunk duration in minutes")
    parser.add_argument("--search_window", type=int, default=DEFAULT_SEARCH_WINDOW, help="Search window in seconds for optimal split points")
    parser.add_argument("--max_subtitle_chars", type=int, default=DEFAULT_MAX_SUBTITLE_CHARS, help="Maximum characters per subtitle line")
    parser.add_argument("--segmentation_strategy", default=DEFAULT_SEGMENTATION_STRATEGY, choices=['spacy', 'whisper'], help="Sentence segmentation strategy")
    
    # Performance and parallel processing
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum parallel workers for chunk processing")
    parser.add_argument("--no_parallel_processing", action="store_true", help="Disable parallel processing")
    
    # Audio preprocessing
    parser.add_argument("--no_audio_preprocessing", action="store_true", help="Disable all audio preprocessing")
    parser.add_argument("--no_normalize_audio", action="store_true", help="Disable volume normalization")
    parser.add_argument("--no_denoise", action="store_true", help="Disable noise reduction")
    parser.add_argument("--noise_reduction_strength", type=float, default=0.5, help="Noise reduction strength (0.1-1.0)")
    parser.add_argument("--use_ai_vocal_separation", action="store_true", help="Use AI vocal separation instead of traditional denoising")
    parser.add_argument("--target_dbfs", type=float, default=-20.0, help="Target dBFS for volume normalization")
    parser.add_argument("--no_speaker_detection", action="store_true", help="Disable speaker change detection")
    parser.add_argument("--min_speaker_duration", type=float, default=2.0, help="Minimum duration between speaker changes (seconds)")
    
    # Output formats
    parser.add_argument("--txt", action="store_true", help="Also save as TXT format")
    parser.add_argument("--vtt", action="store_true", help="Also save as VTT format")
    parser.add_argument("--ass", action="store_true", help="Also save as ASS format")
    parser.add_argument("--json", action="store_true", help="Also save as JSON format")
    
    # File management
    parser.add_argument("--cleanup_preprocessed", action="store_true", help="Delete preprocessed audio files after completion")
    parser.add_argument("--no_save_progress", action="store_true", help="Don't save processing progress")
    parser.add_argument("--word_timestamps", action="store_true", default=True, help="Enable word-level timestamps for accurate segmentation (default: True)")
    
    args = parser.parse_args()
    run_main_workflow(args)
