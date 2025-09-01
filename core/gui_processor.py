"""
GUI Processing Backend
=====================

Integrates existing core functionality with GUI interface.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import logging
import json
import os


class GUIProcessor:
    """Main processor for GUI-initiated operations."""
    
    def __init__(self, settings_manager):
        self.settings_manager = settings_manager
        self.current_process = None
        self.should_stop = False
        self.processing_thread = None

    def _run_subprocess(self, cmd: List[str], log_callback: Callable) -> bool:
        """Generic method to run a subprocess and stream its output."""
        try:
            log_callback(f"Running command: {' '.join(cmd)}")
            
            project_root = Path(__file__).parent.parent
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True,
                cwd=project_root
            )
            
            self.current_process = process

            for line in iter(process.stdout.readline, ''):
                if self.should_stop:
                    log_callback("Termination signal received, stopping subprocess.")
                    process.terminate()
                    break
                
                line = line.strip()
                if line:
                    log_callback(line)
            
            process.stdout.close()
            return_code = process.wait()
            
            if self.should_stop:
                log_callback("Process was stopped by user.")
                return False

            if return_code != 0:
                log_callback(f"Process failed with exit code {return_code}")
                return False
            
            log_callback("Subprocess completed successfully.")
            return True

        except Exception as e:
            error_message = f"Error running subprocess: {e}"
            logging.error(error_message)
            log_callback(error_message)
            return False
        finally:
            self.current_process = None
    
    def process_files(self, files: List[str], settings: Dict[str, Any], log_callback: Callable):
        """Process multiple files with given settings."""
        self.should_stop = False
        total_files = len(files)
        
        try:
            log_callback("Starting processing...")
            
            for i, file_path in enumerate(files):
                if self.should_stop:
                    log_callback("Processing cancelled by user.")
                    return
                
                log_callback(f"--- Processing file {i+1}/{total_files}: {Path(file_path).name} ---")
                
                success = self.process_single_file(file_path, settings, log_callback)
                
                if not success and not self.should_stop:
                    raise Exception(f"Failed to process {Path(file_path).name}")
                
                log_callback(f"--- Finished file {i+1}/{total_files}: {Path(file_path).name} ---")

            if not self.should_stop:
                log_callback("--- All files processed successfully! ---")
            
        except Exception as e:
            error_message = f"Processing failed: {str(e)}"
            logging.error(error_message)
            log_callback(error_message)
    
    def process_single_file(self, file_path: str, settings: Dict[str, Any], 
                          log_callback: Callable) -> bool:
        """Process a single file."""
        try:
            file_name = Path(file_path).stem
            
            log_callback(f"Starting Whisper transcription for {file_name}...")
            
            if not self.run_whisper_transcription(file_path, settings, log_callback):
                return False
            
            if self.should_stop:
                return False
            
            log_callback(f"Whisper transcription completed for {file_name}.")
            
            translation_mode = settings.get('translation_mode', 'free')
            
            if translation_mode in ['local', 'api']:
                log_callback(f"Starting {translation_mode} translation for {file_name}...")
                
                if not self.run_translation(file_name, settings, log_callback):
                     log_callback(f"Translation failed for {file_name}, original subtitles preserved.")
                else:
                     log_callback(f"Translation completed for {file_name}.")

            log_callback(f"File processing completed successfully for {file_name}.")
            return True
            
        except Exception as e:
            error_message = f"Error processing {file_path}: {e}"
            logging.error(error_message)
            log_callback(error_message)
            return False
    
    def run_whisper_transcription(self, file_path: str, settings: Dict[str, Any], log_callback: Callable) -> bool:
        """Run Whisper transcription using main.py."""
        cmd = [
            sys.executable, "src/processing/main.py",
            file_path,
            "--model", settings.get('model', 'medium'),
            "--target_language", settings.get('target_language', 'zh-CN'),
            "--output_format", settings.get('output_format', 'bilingual'),
            "--max_subtitle_chars", str(settings.get('max_chars', 80)),
        ]
        
        if settings.get('source_language', 'auto') != 'auto':
            cmd.extend(["--source_language", settings['source_language']])

        if settings.get('no_audio_preprocessing'):
            cmd.append("--no_audio_preprocessing")
        if settings.get('no_normalize_audio'):
            cmd.append("--no_normalize_audio")
        if settings.get('no_denoise'):
            cmd.append("--no_denoise")
        if settings.get('no_keep_preprocessed'):
            cmd.append("--no_keep_preprocessed")
        if settings.get('translate_during_detection'):
             cmd.append("--translate_during_detection")

        return self._run_subprocess(cmd, log_callback)

    def run_translation(self, project_name: str, settings: Dict[str, Any], log_callback: Callable) -> bool:
        """Run translation using appropriate method."""
        translation_mode = settings.get('translation_mode', 'free')
        
        try:
            if translation_mode == 'local':
                return self.run_local_translation(project_name, settings, log_callback)
            elif translation_mode == 'api':
                return self.run_api_translation(project_name, settings, log_callback)
            else:
                return True
                
        except Exception as e:
            log_callback(f"Translation failed: {e}")
            return False
    
    def run_local_translation(self, project_name: str, settings: Dict[str, Any], log_callback: Callable) -> bool:
        """Run local LLM translation using llm_translate.py."""
        cmd = [
            sys.executable, "src/translation/llm_translate.py",
            project_name,
            "--model", settings.get('local_model', 'qwen2.5:7b'),
            "--chunk-size", str(settings.get('chunk_size', 5))
        ]
        return self._run_subprocess(cmd, log_callback)
    
    def run_api_translation(self, project_name: str, settings: Dict[str, Any], log_callback: Callable) -> bool:
        """Run API translation using relay_api_translate.py."""
        api_key = settings.get('api_key', '')
        base_url = settings.get('base_url', '')
        api_model = settings.get('api_model', 'gemini-2.5-pro')
        
        if not api_key or not base_url:
            api_settings = self.settings_manager.get_api_settings('relay_api')
            api_key = api_key or api_settings.get('api_key', '')
            base_url = base_url or api_settings.get('base_url', '')
        
        if not api_key or not base_url:
            log_callback("Missing API key or base URL for API translation.")
            return False
        
        cmd = [
            sys.executable, "src/translation/relay_api_translate.py",
            project_name,
            "--model", api_model,
            "--api-key", api_key,
            "--base-url", base_url
        ]
        return self._run_subprocess(cmd, log_callback)
    
    def reprocess_chunks(self, project_name: str, chunk_indices: List[int], settings: Dict[str, Any], log_callback: Callable) -> bool:
        """Reprocess specific chunks."""
        chunk_list = ','.join(map(str, chunk_indices))
        
        cmd = [
            sys.executable, "src/processing/reprocess_chunk.py",
            project_name,
            chunk_list,
            "--model", settings.get('model', 'large'),
            "--target_language", settings.get('target_language', 'zh-CN'),
            "--output_format", settings.get('output_format', 'bilingual'),
            "--max_subtitle_chars", str(settings.get('max_chars', 80))
        ]
        
        if settings.get('source_language', 'auto') != 'auto':
            cmd.extend(["--source_language", settings['source_language']])
        
        return self._run_subprocess(cmd, log_callback)
    
    def merge_chunks(self, project_name: str, subfolder: str, log_callback: Callable) -> bool:
        """Merge subtitle chunks from specified subfolder."""
        cmd = [
            sys.executable, "src/processing/merge_subtitles.py",
            project_name
        ]
        
        if subfolder and subfolder != 'original':
            cmd.extend(["--subfolder", subfolder])
        
        return self._run_subprocess(cmd, log_callback)
    
    def get_available_projects(self) -> List[str]:
        """Get list of available projects."""
        output_dir = Path("output_subtitles")
        if not output_dir.exists():
            return []
        
        projects = []
        for item in output_dir.iterdir():
            if item.is_dir():
                projects.append(item.name)
        
        return sorted(projects)
    
    def get_project_chunks(self, project_name: str, subfolder: str = None) -> List[Dict[str, Any]]:
        """Get chunks for a specific project and optional subfolder."""
        project_dir = Path("output_subtitles") / project_name
        if subfolder:
            project_dir = project_dir / subfolder
            
        if not project_dir.exists():
            return []
        
        chunks = []
        chunk_files = list(project_dir.glob("chunk_*.srt"))
        
        for chunk_file in sorted(chunk_files):
            chunk_info = {
                'name': chunk_file.name,
                'path': str(chunk_file),
                'size': chunk_file.stat().st_size,
                'modified': chunk_file.stat().st_mtime,
                'subfolder': subfolder or 'original'
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def get_project_subfolders(self, project_name: str) -> List[str]:
        """Get available translation subfolders for a project."""
        project_dir = Path("output_subtitles") / project_name
        if not project_dir.exists():
            return []
            
        subfolders = []
        
        # Check if original directory has chunks
        if any(project_dir.glob("chunk_*.srt")):
            subfolders.append('original')
        
        # Check for translation subfolders (local_llm, api_llm, etc.)
        for item in project_dir.iterdir():
            if item.is_dir() and any(item.glob("chunk_*.srt")):
                # Map folder names to user-friendly names
                if item.name == "local_llm":
                    subfolders.append("local_llm")
                elif item.name == "api_llm":
                    subfolders.append("api_llm")
                elif item.name == "gemini":  # Legacy support
                    subfolders.append("gemini")
                else:
                    subfolders.append(item.name)
        
        return sorted(subfolders, key=lambda x: (x != 'original', x))
    
    def stop_processing(self):
        """Stop current processing operation."""
        self.should_stop = True
        
        if self.current_process:
            try:
                self.current_process.terminate()
                # Wait a bit for graceful termination
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    self.current_process.kill()
                    
                logging.info("Processing stopped by user")
            except Exception as e:
                logging.error(f"Error stopping process: {e}")
        
        self.current_process = None
    
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self.current_process is not None
    
    def test_api_connection(self, api_settings: Dict[str, Any]) -> bool:
        """Test API connection."""
        try:
            cmd = [
                sys.executable, "src/translation/relay_api_translate.py",
                "test_project",
                "--test",
                "--api-key", api_settings.get('api_key', ''),
                "--base-url", api_settings.get('base_url', '')
            ]
            
            # Remove empty parameters
            cmd = [arg for arg in cmd if arg]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent
            )
            
            return process.returncode == 0
            
        except Exception as e:
            logging.error(f"API connection test failed: {e}")
            return False
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check Ollama server status."""
        try:
            import requests
            
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'running': True,
                    'models': [model['name'] for model in models]
                }
            else:
                return {'running': False, 'models': []}
                
        except Exception as e:
            logging.error(f"Ollama status check failed: {e}")
            return {'running': False, 'models': [], 'error': str(e)}