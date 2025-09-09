#!/usr/bin/env python3
"""
Optimized Chunk Reprocessing Tool
=================================

This tool allows you to reprocess specific audio chunks with improved functionality:
1. Automatically reads chunk timing from chunk_timing.json
2. Can process both original and preprocessed (denoised) audio files
3. Supports batch processing of multiple chunks
4. Maintains perfect time alignment with the original video

Usage Examples:
    # Reprocess single chunk with large model
    python reprocess_chunk.py PVPpractise 1 --model large
    
    # Reprocess multiple chunks
    python reprocess_chunk.py PVPpractise 1,3,5 --model large
    
    # Use preprocessed (denoised) audio if available
    python reprocess_chunk.py PVPpractise 1 --use-preprocessed --model large
    
    # Batch reprocess all chunks
    python reprocess_chunk.py PVPpractise all --model large
"""

import whisper
from deep_translator import GoogleTranslator
import argparse
import os
import srt
from tqdm import tqdm
import datetime
import sys
import json
import re
from pydub import AudioSegment
import glob
from pathlib import Path

# Import shared functions from main.py
import importlib.util
import types

def load_module_from_file(module_name, file_path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load shared functions from main.py
try:
    main_module = load_module_from_file("main", "main.py")
    # Import the functions we need
    load_glossary = main_module.load_glossary
    apply_glossary = main_module.apply_glossary
    translate_text = main_module.translate_text
    finalize_subtitle_text = main_module.finalize_subtitle_text
    structure_and_split_segments = main_module.structure_and_split_segments
    save_subtitles_to_txt = main_module.save_subtitles_to_txt
    save_subtitles_to_vtt = main_module.save_subtitles_to_vtt
    save_subtitles_to_ass = main_module.save_subtitles_to_ass
    save_subtitles_to_json = main_module.save_subtitles_to_json
except Exception as e:
    print(f"Error: Could not import shared functions from main.py: {e}")
    print("Make sure main.py is in the same directory.")
    sys.exit(1)

# Configuration - use dynamic paths
try:
    from src.utils.constants import get_input_dir, get_output_dir, INPUT_DIR_NAME, OUTPUT_DIR_NAME
    INPUT_DIR = INPUT_DIR_NAME  # For backward compatibility
    OUTPUT_DIR = OUTPUT_DIR_NAME  # For backward compatibility
except ImportError:
    INPUT_DIR = "input_audio"
    OUTPUT_DIR = "output_subtitles"

def load_chunk_timing(output_dir):
    """Load chunk timing information from chunk_timing.json."""
    timing_file = os.path.join(output_dir, "chunk_timing.json")
    if not os.path.exists(timing_file):
        print(f"❌ Timing file not found: {timing_file}")
        print("Please run main.py first to generate chunk timing information.")
        return None
    
    try:
        with open(timing_file, 'r', encoding='utf-8') as f:
            timing_data = json.load(f)
        print(f"✅ Loaded timing information for {len(timing_data)} chunks")
        return timing_data
    except Exception as e:
        print(f"❌ Failed to load timing file: {e}")
        return None

def find_best_audio_file(output_dir, chunk_index, use_preprocessed=True):
    """
    Find the best available audio file for a chunk.
    Priority: preprocessed > original chunk file
    Default: Always prefer preprocessed files when available
    """
    raw_chunk_path = Path(output_dir) / "chunks_raw" / f"chunk_{chunk_index}.wav"
    preprocessed_chunk_path = Path(output_dir) / "chunks_preprocessed" / f"chunk_{chunk_index}_preprocessed.wav"

    if use_preprocessed and preprocessed_chunk_path.exists():
        print(f"    📁 Using preprocessed audio: {preprocessed_chunk_path.name}")
        return str(preprocessed_chunk_path)
    
    if raw_chunk_path.exists():
        print(f"    📁 Using original raw audio: {raw_chunk_path.name}")
        if use_preprocessed:
            print(f"    ⚠️  Warning: Preprocessed file not found, falling back to raw audio")
        return str(raw_chunk_path)

    print(f"❌ No audio file found for chunk {chunk_index} in 'chunks_raw' or 'chunks_preprocessed'")
    return None

def process_single_chunk_optimized(chunk_path, chunk_timing, model, args, exact_replace_rules, pre_translate_rules):
    """
    Process a single chunk with optimized logic and precise timing.
    """
    chunk_basename = os.path.basename(chunk_path)
    
    # Convert milliseconds to seconds for display and processing
    start_time_sec = chunk_timing['start_time_ms'] / 1000.0
    end_time_sec = chunk_timing['end_time_ms'] / 1000.0
    duration_sec = chunk_timing['duration_ms'] / 1000.0
    
    print(f"\n🔄 Processing: {chunk_basename}")
    print(f"    ⏱️  Chunk timing: {start_time_sec:.2f}s - {end_time_sec:.2f}s (duration: {duration_sec:.2f}s)")
    
    # Perform transcription
    try:
        transcription = model.transcribe(chunk_path, verbose=True, fp16=False, word_timestamps=True)
    except Exception as e:
        print(f"    ❌ Transcription failed: {e}")
        return []
    
    # Structure and split segments with segmentation strategy
    display_segments = structure_and_split_segments(transcription, args.max_subtitle_chars, args.segmentation_strategy)
    
    subtitles = []
    for segment in tqdm(display_segments, desc=f"    Processing segments", unit="segment"):
        original_text = segment['text'].strip()
        if not original_text:
            continue
        
        # Apply glossary and translation
        processed_text = apply_glossary(original_text, exact_replace_rules)
        
        translated_text = ""
        if (args.translate_during_detection and 
            args.target_language != 'none'):  # Only depends on whether translation is enabled
            source_lang = args.source_language if args.source_language else 'auto'
            translated_text = translate_text(processed_text, args.target_language, pre_translate_rules, source_lang)
        
        final_processed_text = finalize_subtitle_text(processed_text)
        final_translated_text = finalize_subtitle_text(translated_text)
        
        # Determine content based on output format (independent of what was generated)
        content = ""
        if args.output_format == 'bilingual':
            if final_translated_text:
                content = f"{final_processed_text}\n{final_translated_text}"
            else:
                # No translation generated, show source only
                content = final_processed_text
        elif args.output_format == 'target':
            if final_translated_text:
                content = final_translated_text
            else:
                # No translation available, fallback to source
                content = final_processed_text
        else:  # source
            content = final_processed_text
        
        # Apply chunk timing offset
        sub = srt.Subtitle(
            index=0,
            start=datetime.timedelta(seconds=segment['start'] + start_time_sec),
            end=datetime.timedelta(seconds=segment['end'] + start_time_sec),
            content=content
        )
        subtitles.append(sub)
    
    print(f"    ✅ Generated {len(subtitles)} subtitle segments")
    return subtitles

def parse_chunk_list(chunk_spec, available_chunks):
    """
    Parse chunk specification into a list of chunk indices.
    Supports: 1, 1,2,3, 1-3, all
    """
    # Convert available_chunks to a set of chunk indices for easy lookup
    if isinstance(available_chunks, list):
        # If available_chunks is a list of dicts with 'chunk_index' keys
        available_chunk_indices = {str(item['chunk_index']) for item in available_chunks}
    else:
        # If available_chunks is already a dict
        available_chunk_indices = set(available_chunks.keys())
    
    if chunk_spec.lower() == 'all':
        return sorted([int(k) for k in available_chunk_indices])
    
    chunks = []
    for part in chunk_spec.split(','):
        part = part.strip()
        if '-' in part:
            # Range specification like "1-3"
            start, end = map(int, part.split('-'))
            chunks.extend(range(start, end + 1))
        else:
            # Single chunk
            chunks.append(int(part))
    
    # Validate chunks exist
    valid_chunks = []
    for chunk in chunks:
        if str(chunk) in available_chunk_indices:
            valid_chunks.append(chunk)
        else:
            print(f"⚠️  Warning: Chunk {chunk} not found in timing data, skipping")
    
    return sorted(list(set(valid_chunks)))  # Remove duplicates and sort

def regenerate_merged_subtitles(output_dir, args):
    """Regenerate the merged subtitle file from individual chunk files."""
    print("\n🔄 Regenerating merged subtitle files...")
    
    # Find all chunk subtitle files
    chunk_pattern = os.path.join(output_dir, "chunk_*.srt")
    chunk_files = glob.glob(chunk_pattern)
    
    if not chunk_files:
        print("❌ No chunk subtitle files found")
        return
    
    # Sort files by chunk number
    def extract_chunk_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'chunk_(\d+)\.srt', filename)
        return int(match.group(1)) if match else 0
    
    chunk_files.sort(key=extract_chunk_number)
    
    # Load and combine all subtitles
    all_subs = []
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_subs = list(srt.parse(f.read()))
                all_subs.extend(chunk_subs)
                chunk_num = extract_chunk_number(chunk_file)
                print(f"    📄 Loaded chunk {chunk_num}: {len(chunk_subs)} subtitles")
        except Exception as e:
            print(f"    ❌ Failed to load {chunk_file}: {e}")
    
    if not all_subs:
        print("❌ No subtitles loaded")
        return
    
    # Sort and re-index
    all_subs.sort(key=lambda x: x.start)
    for i, sub in enumerate(all_subs):
        sub.index = i + 1
    
    # Save merged files
    base_name = os.path.basename(output_dir)
    
    # Save SRT
    merged_srt_path = os.path.join(output_dir, f"{base_name}_merged.srt")
    with open(merged_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(all_subs))
    print(f"    ✅ Saved merged SRT: {merged_srt_path}")
    
    # Save other formats if requested
    if args.txt or args.all_formats:
        merged_txt_path = os.path.join(output_dir, f"{base_name}_merged.txt")
        save_subtitles_to_txt(all_subs, merged_txt_path)
        print(f"    📝 Saved merged TXT: {merged_txt_path}")
    
    if args.vtt or args.all_formats:
        merged_vtt_path = os.path.join(output_dir, f"{base_name}_merged.vtt")
        save_subtitles_to_vtt(all_subs, merged_vtt_path)
        print(f"    🌐 Saved merged VTT: {merged_vtt_path}")
    
    if args.ass or args.all_formats:
        merged_ass_path = os.path.join(output_dir, f"{base_name}_merged.ass")
        save_subtitles_to_ass(all_subs, merged_ass_path)
        print(f"    🎬 Saved merged ASS: {merged_ass_path}")
    
    if args.json or args.all_formats:
        merged_json_path = os.path.join(output_dir, f"{base_name}_merged.json")
        save_subtitles_to_json(all_subs, merged_json_path)
        print(f"    📊 Saved merged JSON: {merged_json_path}")
    
    print(f"✅ Merged {len(all_subs)} subtitles from {len(chunk_files)} chunks")

def main():
    parser = argparse.ArgumentParser(description="Optimized chunk reprocessing tool")
    parser.add_argument("project_name", help="Project name (e.g., 'PVPpractise')")
    parser.add_argument("chunks", help="Chunk(s) to process: single (1), multiple (1,2,3), range (1-3), or 'all'")
    
    # Model and language parameters
    parser.add_argument("--model", default="large", help="Whisper model size (default: large)")
    parser.add_argument("--target_language", default="none", help="Target language for translation")
    parser.add_argument("--source_language", help="Source language (auto-detect if not specified)")
    parser.add_argument("--output_format", default="bilingual", choices=['bilingual', 'source', 'target'],
                       help="Output format: bilingual, source only, or target only")
    
    # Processing options
    parser.add_argument("--max_subtitle_chars", type=int, default=80, help="Maximum characters per subtitle line")
    parser.add_argument("--segmentation_strategy", default="rule_based", 
                       choices=['rule_based', 'whisper', 'hybrid'],
                       help="Segmentation strategy: rule_based, whisper, or hybrid")
    parser.add_argument("--translate_during_detection", action="store_true",
                       help="Enable translation during processing")
    parser.add_argument("--use-preprocessed", action="store_true",
                       help="Use preprocessed (denoised/normalized) audio files if available")
    
    # Output format options
    parser.add_argument("--txt", action="store_true", help="Also save in TXT format")
    parser.add_argument("--vtt", action="store_true", help="Also save in VTT format")
    parser.add_argument("--ass", action="store_true", help="Also save in ASS format")
    parser.add_argument("--json", action="store_true", help="Also save in JSON format")
    parser.add_argument("--all-formats", action="store_true", help="Save in all formats")
    
    # Special modes
    parser.add_argument("--regenerate-merged", action="store_true",
                       help="Only regenerate merged subtitle files without reprocessing chunks")
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = os.path.join(OUTPUT_DIR, args.project_name)
    if not os.path.exists(output_dir):
        print(f"❌ Project directory not found: {output_dir}")
        print("Please run main.py first to create the project.")
        sys.exit(1)
    
    # Load glossary
    exact_replace_rules, pre_translate_rules = load_glossary(Path(output_dir))
    
    # Special mode: only regenerate merged files
    if args.regenerate_merged:
        regenerate_merged_subtitles(output_dir, args)
        return
    
    # Load chunk timing information
    chunk_timing = load_chunk_timing(output_dir)
    if not chunk_timing:
        sys.exit(1)
    
    # Parse chunk specification
    chunks_to_process = parse_chunk_list(args.chunks, chunk_timing)
    if not chunks_to_process:
        print("❌ No valid chunks to process")
        sys.exit(1)
    
    print(f"🎯 Will process {len(chunks_to_process)} chunk(s): {chunks_to_process}")
    
    # Load Whisper model
    print(f"🤖 Loading Whisper model: {args.model}")
    try:
        model = whisper.load_model(args.model)
        print(f"    ✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Process each chunk
    processed_count = 0
    for chunk_index in chunks_to_process:
        print(f"\n{'='*60}")
        print(f"📦 Processing Chunk {chunk_index}/{len(chunk_timing)} ")
        print(f"{'='*60}")
        
        # Find the best audio file for this chunk
        chunk_path = find_best_audio_file(output_dir, chunk_index, args.use_preprocessed)
        if not chunk_path:
            continue
        
        # Get timing information for this chunk
        chunk_timing_info = None
        for timing_item in chunk_timing:
            if timing_item['chunk_index'] == chunk_index:
                chunk_timing_info = timing_item
                break
        
        if not chunk_timing_info:
            print(f"    ❌ No timing information found for chunk {chunk_index}")
            continue
        
        # Process the chunk
        try:
            subtitles = process_single_chunk_optimized(
                chunk_path, chunk_timing_info, model, args,
                exact_replace_rules, pre_translate_rules
            )
            
            if not subtitles:
                print(f"    ⚠️  No subtitles generated for chunk {chunk_index}")
                continue
            
            # Save individual chunk file
            chunk_srt_path = os.path.join(output_dir, f"chunk_{chunk_index}.srt")
            with open(chunk_srt_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(subtitles))
            print(f"    💾 Saved: {chunk_srt_path}")
            
            # Save other formats if requested
            if args.txt or args.all_formats:
                chunk_txt_path = os.path.join(output_dir, f"chunk_{chunk_index}.txt")
                save_subtitles_to_txt(subtitles, chunk_txt_path)
                print(f"    📝 Saved TXT: {chunk_txt_path}")
            
            if args.vtt or args.all_formats:
                chunk_vtt_path = os.path.join(output_dir, f"chunk_{chunk_index}.vtt")
                save_subtitles_to_vtt(subtitles, chunk_vtt_path)
                print(f"    🌐 Saved VTT: {chunk_vtt_path}")
            
            if args.ass or args.all_formats:
                chunk_ass_path = os.path.join(output_dir, f"chunk_{chunk_index}.ass")
                save_subtitles_to_ass(subtitles, chunk_ass_path)
                print(f"    🎬 Saved ASS: {chunk_ass_path}")
            
            if args.json or args.all_formats:
                chunk_json_path = os.path.join(output_dir, f"chunk_{chunk_index}.json")
                save_subtitles_to_json(subtitles, chunk_json_path)
                print(f"    📊 Saved JSON: {chunk_json_path}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"    ❌ Failed to process chunk {chunk_index}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"📋 Processing Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {processed_count}/{len(chunks_to_process)} chunks")
    
    if processed_count > 0:
        # Regenerate merged subtitle files
        regenerate_merged_subtitles(output_dir, args)
        
        print(f"\n🎉 Chunk reprocessing completed!")
        print(f"📁 Results saved to: {output_dir}")
    else:
        print(f"\n❌ No chunks were successfully processed")

if __name__ == "__main__":
    main()