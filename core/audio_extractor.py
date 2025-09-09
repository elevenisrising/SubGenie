"""
Audio Extractor - Standalone Audio Extraction Utility
=====================================================

Simple video-to-audio conversion tool using FFmpeg.
"""

import subprocess
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable


class AudioExtractor:
    """Standalone audio extraction utility."""
    
    def __init__(self):
        self.supported_video_formats = [
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', 
            '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'
        ]
        self.supported_audio_formats = ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg']
        
    def extract_single_file(self, input_file: str, output_dir: str, 
                          audio_format: str = "wav", quality: str = "medium") -> str:
        """
        Extract audio from a single video file.
        
        Args:
            input_file: Path to input video file
            output_dir: Directory to save audio file
            audio_format: Output audio format (wav, mp3, m4a, flac)
            quality: Audio quality (low, medium, high)
            
        Returns:
            Path to extracted audio file
        """
        # Validate input file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        if input_path.suffix.lower() not in self.supported_video_formats:
            raise ValueError(f"Unsupported video format: {input_path.suffix}")
        
        # Validate audio format
        if audio_format not in self.supported_audio_formats:
            raise ValueError(f"Unsupported audio format: {audio_format}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{input_path.stem}.{audio_format}"
        output_file = output_path / output_filename
        
        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(input_file, str(output_file), audio_format, quality)
        
        # Execute extraction
        try:
            logging.info(f"Extracting audio: {input_path.name} -> {output_filename}")
            
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            # Verify output file was created
            if not output_file.exists():
                raise RuntimeError("Output file was not created")
                
            logging.info(f"Successfully extracted: {output_filename}")
            return str(output_file)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed: {e.stderr or e.stdout or str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")
    
    def extract_batch(self, input_files: List[str], output_dir: str, 
                     audio_format: str = "wav", quality: str = "medium",
                     progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Extract audio from multiple video files.
        
        Args:
            input_files: List of input video file paths
            output_dir: Directory to save audio files
            audio_format: Output audio format
            quality: Audio quality
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of extracted audio file paths
        """
        extracted_files = []
        total_files = len(input_files)
        
        for i, input_file in enumerate(input_files):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(i, total_files, os.path.basename(input_file))
                
                # Extract audio
                output_file = self.extract_single_file(
                    input_file, output_dir, audio_format, quality
                )
                extracted_files.append(output_file)
                
            except Exception as e:
                logging.error(f"Failed to extract {input_file}: {e}")
                # Continue with other files instead of stopping
                continue
        
        return extracted_files
    
    def _build_ffmpeg_command(self, input_file: str, output_file: str, 
                            audio_format: str, quality: str) -> List[str]:
        """Build FFmpeg command based on format and quality settings."""
        
        cmd = ['ffmpeg', '-i', input_file, '-vn']  # -vn = no video
        
        # Format-specific encoding settings
        if audio_format == 'wav':
            cmd.extend(['-acodec', 'pcm_s16le'])
            
        elif audio_format == 'mp3':
            cmd.extend(['-acodec', 'libmp3lame'])
            # Quality settings for MP3
            if quality == 'low':
                cmd.extend(['-b:a', '128k'])
            elif quality == 'medium':
                cmd.extend(['-b:a', '192k'])
            elif quality == 'high':
                cmd.extend(['-b:a', '320k'])
                
        elif audio_format == 'm4a':
            cmd.extend(['-acodec', 'aac'])
            # Quality settings for AAC
            if quality == 'low':
                cmd.extend(['-b:a', '128k'])
            elif quality == 'medium':
                cmd.extend(['-b:a', '192k'])
            elif quality == 'high':
                cmd.extend(['-b:a', '256k'])
                
        elif audio_format == 'flac':
            cmd.extend(['-acodec', 'flac'])
            # FLAC compression levels
            if quality == 'low':
                cmd.extend(['-compression_level', '0'])
            elif quality == 'medium':
                cmd.extend(['-compression_level', '5'])
            elif quality == 'high':
                cmd.extend(['-compression_level', '8'])
                
        elif audio_format == 'aac':
            cmd.extend(['-acodec', 'aac'])
            # Quality settings for AAC
            if quality == 'low':
                cmd.extend(['-b:a', '128k'])
            elif quality == 'medium':
                cmd.extend(['-b:a', '192k'])
            elif quality == 'high':
                cmd.extend(['-b:a', '256k'])
                
        elif audio_format == 'ogg':
            cmd.extend(['-acodec', 'libvorbis'])
            # Quality settings for Vorbis
            if quality == 'low':
                cmd.extend(['-q:a', '3'])
            elif quality == 'medium':
                cmd.extend(['-q:a', '6'])
            elif quality == 'high':
                cmd.extend(['-q:a', '9'])
        
        # Add output file and overwrite flag
        cmd.extend(['-y', output_file])
        
        return cmd
    
    def get_media_info(self, input_file: str) -> Dict:
        """
        Get basic information about a media file using FFprobe.
        
        Args:
            input_file: Path to media file
            
        Returns:
            Dictionary with media information
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', input_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            import json
            data = json.loads(result.stdout)
            
            # Extract useful information
            info = {
                'duration': 0,
                'audio_streams': [],
                'video_streams': []
            }
            
            if 'format' in data:
                info['duration'] = float(data['format'].get('duration', 0))
                
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    info['audio_streams'].append({
                        'codec': stream.get('codec_name'),
                        'sample_rate': stream.get('sample_rate'),
                        'channels': stream.get('channels'),
                        'duration': stream.get('duration')
                    })
                elif stream.get('codec_type') == 'video':
                    info['video_streams'].append({
                        'codec': stream.get('codec_name'),
                        'width': stream.get('width'),
                        'height': stream.get('height'),
                        'duration': stream.get('duration')
                    })
            
            return info
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Could not get media info for {input_file}: {e}")
            return {'duration': 0, 'audio_streams': [], 'video_streams': []}
    
    def is_valid_video_file(self, file_path: str) -> bool:
        """Check if file is a supported video format."""
        if not os.path.exists(file_path):
            return False
            
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_video_formats


def main():
    """Command-line interface for audio extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract audio from video files")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", default="./extracted_audio", 
                       help="Output directory (default: ./extracted_audio)")
    parser.add_argument("-f", "--format", default="wav", 
                       choices=['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg'],
                       help="Output audio format (default: wav)")
    parser.add_argument("-q", "--quality", default="medium",
                       choices=['low', 'medium', 'high'],
                       help="Audio quality (default: medium)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create extractor
    extractor = AudioExtractor()
    
    # Determine input files
    input_path = Path(args.input)
    input_files = []
    
    if input_path.is_file():
        if extractor.is_valid_video_file(str(input_path)):
            input_files = [str(input_path)]
        else:
            logging.error(f"Unsupported file format: {input_path}")
            return 1
    elif input_path.is_dir():
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and extractor.is_valid_video_file(str(file_path)):
                input_files.append(str(file_path))
    else:
        logging.error(f"Input path not found: {input_path}")
        return 1
    
    if not input_files:
        logging.error("No valid video files found")
        return 1
    
    # Extract audio
    logging.info(f"Found {len(input_files)} video file(s)")
    
    def progress_callback(current, total, filename):
        logging.info(f"Processing ({current+1}/{total}): {filename}")
    
    try:
        extracted = extractor.extract_batch(
            input_files, args.output, args.format, args.quality, progress_callback
        )
        
        logging.info(f"Successfully extracted {len(extracted)} audio file(s)")
        logging.info(f"Output directory: {args.output}")
        
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())