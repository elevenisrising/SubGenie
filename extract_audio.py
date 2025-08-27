#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Audio Extractor
A tool to quickly extract audio tracks from video files like MP4, AVI, MKV, etc., 
with support for various audio output formats.
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import json
import time
from tqdm import tqdm

# =============================================================================
# Configurable Parameters
# =============================================================================
# The following parameters can be adjusted via command-line arguments:
#
# 1. Audio Quality Parameters:
#    - audio_bitrate: Audio bitrate (kbps)
#    - audio_channels: Number of audio channels (1=mono, 2=stereo)
#    - audio_sample_rate: Audio sample rate (Hz)
#
# 2. Output Format Parameters:
#    - output_format: Output audio format (mp3, wav, m4a, aac, etc.)
#    - output_dir: Output directory
#
# 3. Processing Parameters:
#    - overwrite: Whether to overwrite existing files
#    - verbose: Whether to display detailed processing information
# =============================================================================

# Default Configuration
DEFAULT_BITRATE = 192  # kbps
DEFAULT_CHANNELS = 2   # stereo
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_FORMAT = "mp3"
INPUT_VIDEO_DIR = "input_video"
OUTPUT_AUDIO_DIR = "input_audio"

# Supported audio formats and their corresponding ffmpeg parameters
AUDIO_FORMATS = {
    "mp3": {
        "codec": "libmp3lame",
        "extension": "mp3",
        "description": "MP3 format (recommended, small file size, good compatibility)"
    },
    "wav": {
        "codec": "pcm_s16le",
        "extension": "wav",
        "description": "WAV format (lossless, large file size)"
    },
    "m4a": {
        "codec": "aac",
        "extension": "m4a",
        "description": "M4A format (AAC encoded, small file size)"
    },
    "aac": {
        "codec": "aac",
        "extension": "aac",
        "description": "AAC format (high quality, small file size)"
    },
    "flac": {
        "codec": "flac",
        "extension": "flac",
        "description": "FLAC format (lossless compressed)"
    },
    "ogg": {
        "codec": "libvorbis",
        "extension": "ogg",
        "description": "OGG format (open source, small file size)"
    }
}

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0

def extract_audio_with_progress(video_path, output_path, format_info, bitrate, channels, sample_rate, verbose=False):
    """Extract audio with a progress bar."""
    try:
        duration = get_video_duration(video_path)
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # No video
            "-acodec", format_info["codec"],
            "-ar", str(sample_rate),
            "-ac", str(channels)
        ]
        
        if format_info["codec"] in ["libmp3lame", "aac"]:
            cmd.extend(["-b:a", f"{bitrate}k"])
        
        cmd.append(str(output_path))
        
        if verbose:
            print(f"🔧 Executing command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            universal_newlines=True
        )
        
        with tqdm(total=100, desc="🎵 Extracting Audio", unit="%") as pbar:
            if duration > 0:
                for line in process.stdout:
                    if "time=" in line:
                        time_str = line.split("time=")[1].split(" ")[0]
                        h, m, s = map(float, time_str.split(':'))
                        elapsed_time = h * 3600 + m * 60 + s
                        progress = min(100, (elapsed_time / duration) * 100)
                        pbar.n = int(progress)
                        pbar.refresh()
            else: # Fallback for when duration is not available
                pbar.set_description("🎵 Extracting Audio (duration unknown)")
                process.wait()
                pbar.n = 100

        return process.returncode == 0

    except Exception as e:
        print(f"❌ An error occurred during extraction: {e}")
        return False

def format_duration(seconds):
    """Formats duration for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"

def main():
    parser = argparse.ArgumentParser(
        description="Quickly extract audio tracks from video files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Usage Examples:
  # Basic usage (extract to MP3)
  python extract_audio.py my_video.mp4
  
  # Specify output format and quality
  python extract_audio.py my_video.mp4 --format wav --bitrate 320
  
  # Batch process all videos in a directory
  python extract_audio.py --batch --format m4a
  
  # High-quality output
  python extract_audio.py my_video.mp4 --format flac --channels 2 --sample-rate 48000
'''
    )
    
    parser.add_argument("filename", nargs="?", help="Name of the video file in the input directory.")
    parser.add_argument("--batch", action="store_true", help="Batch process all supported video files in the input directory.")
    parser.add_argument("--input-dir", default=INPUT_VIDEO_DIR, help=f"Input video directory (default: {INPUT_VIDEO_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_AUDIO_DIR, help=f"Output audio directory (default: {OUTPUT_AUDIO_DIR})")
    parser.add_argument("--format", choices=list(AUDIO_FORMATS.keys()), default=DEFAULT_FORMAT, help=f"Output audio format (default: {DEFAULT_FORMAT})")
    parser.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE, help=f"Audio bitrate in kbps (default: {DEFAULT_BITRATE})")
    parser.add_argument("--channels", type=int, choices=[1, 2], default=DEFAULT_CHANNELS, help=f"Number of audio channels (default: {DEFAULT_CHANNELS})")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help=f"Audio sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--verbose", action="store_true", help="Display detailed processing information.")
    parser.add_argument("--list-formats", action="store_true", help="List all supported audio formats.")
    
    args = parser.parse_args()
    
    if args.list_formats:
        print("🎵 Supported Audio Formats:")
        for fmt, info in AUDIO_FORMATS.items():
            print(f"  - {fmt}: {info['description']}")
        return
    
    if not check_ffmpeg():
        print("❌ Error: ffmpeg not found. Please install it and ensure it's in your system's PATH.")
        print("  - Windows: choco install ffmpeg -y")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux: sudo apt install ffmpeg")
        sys.exit(1)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    format_info = AUDIO_FORMATS[args.format]
    
    if args.batch:
        video_files = [f for f in input_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']]
        if not video_files:
            print(f"❌ No supported video files found in '{input_dir}'.")
            return
        
        print(f"🎬 Found {len(video_files)} video(s). Starting batch processing...")
        for video_file in video_files:
            print(f"\n📹 Processing: {video_file.name}")
            process_single_video(video_file, output_dir, format_info, args)
            
    elif args.filename:
        video_path = input_dir / args.filename
        if not video_path.exists():
            print(f"❌ Error: File '{args.filename}' not found in '{input_dir}'.")
            sys.exit(1)
        
        process_single_video(video_path, output_dir, format_info, args)
        
    else:
        print("❌ Error: Please specify a video filename or use the --batch flag.")
        parser.print_help()
        sys.exit(1)

def process_single_video(video_path, output_dir, format_info, args):
    """Processes a single video file."""
    try:
        duration = get_video_duration(video_path)
        duration_str = format_duration(duration) if duration > 0 else "Unknown"
        print(f"    📊 Duration: {duration_str}")
        
        output_filename = f"{video_path.stem}.{format_info['extension']}"
        output_path = output_dir / output_filename
        
        if output_path.exists() and not args.overwrite:
            print(f"    ⚠️  File already exists, skipping. (Use --overwrite to force)")
            return
        
        print(f"    🎵 Extracting audio to: {output_path.name}")
        
        if extract_audio_with_progress(video_path, output_path, format_info, 
                        args.bitrate, args.channels, args.sample_rate, args.verbose):
            output_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"    ✅ Done! Output size: {output_size:.1f} MB")
        else:
            print(f"    ❌ Audio extraction failed.")
            
    except Exception as e:
        print(f"    ❌ An error occurred while processing: {e}")

if __name__ == "__main__":
    main()