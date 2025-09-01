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
from typing import Any, Dict, List

import spacy

# --- spaCy Model Loading ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy English model 'en_core_web_sm' not found.")
    logging.error(f"Please run 'D:\\Anaconda\\envs\\asr-env\\python.exe -m spacy download en_core_web_sm' to install it.")
    sys.exit(1)
import numpy as np
import srt
import torch
import whisper
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent
from tqdm import tqdm

# --- Configuration ---
LOG_FORMAT = '%(asctime)s - [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# --- Constants ---
GLOSSARY_FILE_NAME = "src/utils/glossary.json"
CACHE_DIR = ".cache"
CACHE_EXPIRY_DAYS = 7
MERGE_WORD_THRESHOLD = 5
FINAL_PUNCTUATION = '.?!。？！…'
CHUNK_TARGET_DURATION_MIN = 30  # Default 30 minutes
CHUNK_SEARCH_WINDOW_SEC = 120   # 2 minutes search window

# --- Cache Management ---
def get_audio_hash(audio_path: Path) -> str:
    """Calculate SHA256 hash of an audio file for caching."""
    try:
        with open(audio_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "no_hash"

def get_cache_path(audio_hash: str, model_name: str, args_hash: str) -> Path:
    """Generate cache file path."""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{audio_hash}_{model_name}_{args_hash}.pkl"

def is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(cache_path.stat().st_mtime)
    return file_age.days < CACHE_EXPIRY_DAYS

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
    output_dir = Path("output_subtitles") / base_name
    
    # Check if input is a project name (look in input_audio directory)
    input_audio_dir = Path("input_audio")
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

def denoise_audio(audio: AudioSegment, strength: float = 0.5) -> AudioSegment:
    """Apply simple noise reduction to audio."""
    samples = np.array(audio.get_array_of_samples())
    noise_floor = np.percentile(np.abs(samples), 10)
    threshold = noise_floor * (1 + strength)
    denoised_samples = np.where(np.abs(samples) < threshold, 0, samples).astype(np.int16)
    return audio._spawn(denoised_samples)

def detect_speaker_changes(audio: AudioSegment, min_duration_sec: float) -> List[float]:
    """Detect potential speaker changes in audio."""
    samples = np.array(audio.get_array_of_samples())
    frame_rate = audio.frame_rate
    window_size = int(0.1 * frame_rate)  # 100ms windows
    
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
            if relative_change > 0.5:
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
    """Split audio into chunks and create timing information."""
    logging.info("Loading audio for splitting...")
    audio = AudioSegment.from_file(audio_path)
    duration_min = len(audio) / 60000
    
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
        
        pbar = tqdm(total=len(audio), desc="Splitting audio", unit="ms")
        
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
            
            pbar.update(end_ms - start_ms)
            start_ms = end_ms
            chunk_index += 1
        
        pbar.close()
    
    # Save timing information
    with open(paths["timing_info"], 'w', encoding='utf-8') as f:
        json.dump(timing_info, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Audio split into {len(timing_info)} chunks.")
    logging.info(f"Timing information saved to: {paths['timing_info']}")
    
    return timing_info

# --- Subtitle Processing (spaCy Implementation) ---
def structure_and_split_segments(transcription_result: Dict, max_chars: int) -> List[Dict]:
    """Structures subtitles using spaCy with a two-step process for natural segmentation."""
    all_words = []
    for seg in transcription_result.get("segments", []):
        if 'words' in seg and seg['words']:
            all_words.extend(seg['words'])
    
    if not all_words:
        logging.warning("Word-level timestamps not found. spaCy segmentation requires it.")
        return []

    # Create a continuous text string and a map from character index to word index
    full_text = ""
    char_to_word_map = {}
    current_char = 0
    for i, word_info in enumerate(all_words):
        word = word_info['word'].strip()
        start_char = len(full_text)
        full_text += word + " "
        end_char = len(full_text)
        for char_idx in range(start_char, end_char):
            char_to_word_map[char_idx] = i

    doc = nlp(full_text)
    final_segments = []

    # Step 1: Process sentences from spaCy's default sentence segmenter
    for sent in doc.sents:
        if not sent.text.strip():
            continue

        # If the sentence is short enough, process it directly
        if len(sent.text) <= max_chars:
            add_segment_from_span(sent, all_words, char_to_word_map, final_segments)
        else:
            # Step 2: If a sentence is too long, apply custom splitting logic
            split_spans = split_long_sentence(sent, max_chars)
            for sub_span in split_spans:
                add_segment_from_span(sub_span, all_words, char_to_word_map, final_segments)

    return final_segments

def add_segment_from_span(span: spacy.tokens.span.Span, all_words: List[Dict], char_map: Dict, final_segments: List):
    """Helper function to create and add a subtitle segment from a spaCy span."""
    start_char_idx = span.start_char
    end_char_idx = min(span.end_char - 1, max(char_map.keys()))

    start_word_idx = char_map.get(start_char_idx)
    end_word_idx = char_map.get(end_char_idx)

    if start_word_idx is None or end_word_idx is None:
        return

    sent_words = all_words[start_word_idx : end_word_idx + 1]
    if not sent_words:
        return

    final_segments.append({
        'text': span.text.strip(),
        'start': sent_words[0]['start'],
        'end': sent_words[-1]['end']
    })

def split_long_sentence(sent: spacy.tokens.span.Span, max_chars: int) -> List[spacy.tokens.span.Span]:
    """Splits a long spaCy sentence at natural boundaries like commas or conjunctions."""
    sub_sentences = []
    start_token_idx = sent.start

    while start_token_idx < sent.end:
        current_len = 0
        last_safe_split_idx = -1
        current_token_idx = start_token_idx

        while current_token_idx < sent.end:
            token = sent.doc[current_token_idx]
            if current_len + len(token.text_with_ws) > max_chars:
                break
            
            current_len += len(token.text_with_ws)
            
            # Split BEFORE coordinating conjunctions for better flow
            if (token.pos_ == 'CCONJ' or 
                token.text.lower() in ['but', 'and', 'or', 'so']):
                last_safe_split_idx = current_token_idx
            # Split AFTER other connectors and punctuation
            elif (token.text == ',' or 
                  token.pos_ == 'SCONJ' or  # because, when, if, although
                  token.text.lower() in ['then', 'also', 'however']):
                last_safe_split_idx = current_token_idx + 1
            
            current_token_idx += 1

        if last_safe_split_idx != -1 and last_safe_split_idx > start_token_idx:
            sub_sentences.append(sent.doc[start_token_idx:last_safe_split_idx])
            start_token_idx = last_safe_split_idx
        else:
            end_idx = min(current_token_idx, sent.end)
            if end_idx == start_token_idx:
                 end_idx += 1
            sub_sentences.append(sent.doc[start_token_idx:end_idx])
            start_token_idx = end_idx
            
    return sub_sentences

# --- File Saving Utilities ---
def save_subtitles_to_srt(subs: List[srt.Subtitle], path: str):
    """Save subtitles in SRT format."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

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
    logging.info(f"Processing chunk: {chunk_path.name}")
    
    preprocessed_files_to_keep = []
    
    # 1. Audio Preprocessing
    audio = AudioSegment.from_file(chunk_path)
    audio_to_transcribe = str(chunk_path)
    
    if not args.no_audio_preprocessing:
        logging.info("Applying audio preprocessing...")
        
        if not args.no_normalize_audio:
            audio = normalize_audio_volume(audio, args.target_dbfs)
        
        if not args.no_denoise:
            audio = denoise_audio(audio, args.noise_reduction_strength)
        
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
    args_hash = hashlib.md5(f"{args.source_language}_{args.max_subtitle_chars}".encode()).hexdigest()[:8]
    cache_path = get_cache_path(audio_hash, args.model, args_hash)
    
    if is_cache_valid(cache_path):
        logging.info(f"Loading transcription from cache: {cache_path.name}")
        result = load_from_cache(cache_path)
    else:
        logging.info("Transcribing with Whisper model...")
        try:
            result = model.transcribe(
                audio_to_transcribe,
                language=args.source_language,
                word_timestamps=True,
                fp16=False
            )
            save_to_cache(cache_path, result)
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return [], len(audio) / 1000.0, preprocessed_files_to_keep
    
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
    
    if not result or 'segments' not in result:
        logging.warning("No transcription result or segments found")
        return [], len(audio) / 1000.0, preprocessed_files_to_keep
    
    # 3. Structure and format subtitles
    structured_segments = structure_and_split_segments(result, args.max_subtitle_chars)
    logging.info(f"Structured segments count: {len(structured_segments)}")
    
    subtitles = []
    for seg in structured_segments:
        original = seg['text'].strip()
        logging.info(f"Processing segment text: '{original}' (length: {len(original)})")
        if not original:
            logging.info("Skipping empty segment text")
            continue
        
        # Apply glossary
        processed = apply_glossary(original, glossary["exact_replace"])
        
        # Translate if needed
        translated = ""
        if args.translate_during_detection and args.output_format != 'source':
            translated = translate_text(
                processed, 
                args.target_language, 
                glossary["pre_translate"], 
                args.source_language or 'auto'
            )
        
        # Finalize text
        final_processed = finalize_subtitle_text(processed)
        final_translated = finalize_subtitle_text(translated)
        
        # Create content based on output format
        if args.output_format == 'bilingual':
            content = f"{final_processed}\n{final_translated}"
        elif args.output_format == 'target':
            content = final_translated or final_processed
        else:  # source
            content = final_processed
        
        subtitle = srt.Subtitle(
            index=0,  # Will be re-indexed later
            start=datetime.timedelta(seconds=seg['start']),
            end=datetime.timedelta(seconds=seg['end']),
            content=content
        )
        subtitles.append(subtitle)
    
    actual_duration_sec = len(audio) / 1000.0
    return subtitles, actual_duration_sec, preprocessed_files_to_keep

def process_chunk_with_offset(task_info: dict) -> dict:
    """Wrapper for parallel execution that loads model inside thread."""
    chunk_index = task_info['index']
    chunk_path = task_info['path']
    args = task_info['args']
    glossary = task_info['glossary']
    timing_info = task_info['timing_info']
    
    try:
        # Load model in thread for thread safety
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(args.model, device=device)
        
        subs, actual_duration, preprocessed_files = process_single_chunk(
            chunk_path, model, args, glossary, timing_info
        )
        
        return {
            "index": chunk_index,
            "subs": subs,
            "actual_duration": actual_duration,
            "preprocessed_files": preprocessed_files,
            "status": "ok"
        }
    except Exception as e:
        logging.error(f"Failed to process chunk {chunk_index}: {e}")
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
        cmd = [
            'ffmpeg', '-i', str(paths["input_file"]),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', str(paths["extracted_audio"])
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info("Audio extraction completed.")
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
    
    # Create chunk start times map for precise timing
    chunk_start_times = {}
    cumulative_time = 0.0
    for timing in timing_info:
        chunk_index = timing['chunk_index']
        chunk_start_times[chunk_index] = cumulative_time
        cumulative_time += timing['duration_ms'] / 1000.0
    
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
    
    # Process chunks in parallel
    all_results = []
    max_workers = 1 if args.no_parallel_processing else min(args.max_workers, len(tasks))
    
    logging.info(f"Starting processing with {max_workers} worker(s)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk_with_offset, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing chunks"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                logging.error(f"A task failed with exception: {e}")
    
    # Sort results by chunk index
    all_results.sort(key=lambda r: r['index'])
    
    # Combine results and apply timing offsets
    all_subs = []
    processed_chunks = []
    all_preprocessed_files = []
    
    for result in all_results:
        if result['status'] == 'ok' and result['subs']:
            chunk_index = result['index']
            start_time_sec = chunk_start_times.get(chunk_index, 0.0)
            time_offset = datetime.timedelta(seconds=start_time_sec)
            
            # Apply time offset to each subtitle
            for sub in result['subs']:
                sub.start += time_offset
                sub.end += time_offset
                all_subs.append(sub)
            
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
    if not args.no_keep_preprocessed:
        for file_path in all_preprocessed_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logging.info(f"Cleaned up preprocessed file: {file_path.name}")
            except Exception as e:
                logging.warning(f"Failed to cleanup {file_path}: {e}")
    else:
        logging.info(f"Kept {len(all_preprocessed_files)} preprocessed files for reprocessing")
    
    logging.info(f"✨ Processing completed successfully! ✨")
    logging.info(f"Final subtitles saved to: {paths['final_srt']}")
    logging.info(f"Total subtitles generated: {len(all_subs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subtitles from audio/video files with advanced features.")
    
    # Core arguments
    parser.add_argument("filename", help="Input audio or video file")
    parser.add_argument("--model", default="medium", help="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)")
    parser.add_argument("--target_language", default="zh-CN", help="Target language for translation")
    parser.add_argument("--source_language", default=None, help="Source language code (auto-detect if not set)")
    parser.add_argument("--output_format", default="bilingual", choices=['bilingual', 'source', 'target'], help="Subtitle output format")
    parser.add_argument("--translate_during_detection", action="store_true", help="Enable translation during processing")
    
    # Chunking and processing
    parser.add_argument("--long_audio_threshold", type=int, default=15, help="Minutes threshold for chunking audio")
    parser.add_argument("--chunk_duration", type=int, default=30, help="Target chunk duration in minutes")
    parser.add_argument("--search_window", type=int, default=120, help="Search window in seconds for optimal split points")
    parser.add_argument("--max_subtitle_chars", type=int, default=80, help="Maximum characters per subtitle line")
    
    # Performance and parallel processing
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum parallel workers for chunk processing")
    parser.add_argument("--no_parallel_processing", action="store_true", help="Disable parallel processing")
    
    # Audio preprocessing
    parser.add_argument("--no_audio_preprocessing", action="store_true", help="Disable all audio preprocessing")
    parser.add_argument("--no_normalize_audio", action="store_true", help="Disable volume normalization")
    parser.add_argument("--no_denoise", action="store_true", help="Disable noise reduction")
    parser.add_argument("--noise_reduction_strength", type=float, default=0.5, help="Noise reduction strength (0.1-1.0)")
    parser.add_argument("--target_dbfs", type=float, default=-20.0, help="Target dBFS for volume normalization")
    parser.add_argument("--no_speaker_detection", action="store_true", help="Disable speaker change detection")
    parser.add_argument("--min_speaker_duration", type=float, default=2.0, help="Minimum duration between speaker changes (seconds)")
    
    # Output formats
    parser.add_argument("--txt", action="store_true", help="Also save as TXT format")
    parser.add_argument("--vtt", action="store_true", help="Also save as VTT format")
    parser.add_argument("--ass", action="store_true", help="Also save as ASS format")
    parser.add_argument("--json", action="store_true", help="Also save as JSON format")
    
    # File management
    parser.add_argument("--no_keep_preprocessed", action="store_true", help="Don't keep preprocessed audio files")
    parser.add_argument("--no_save_progress", action="store_true", help="Don't save processing progress")
    
    args = parser.parse_args()
    run_main_workflow(args)