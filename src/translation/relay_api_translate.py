#!/usr/bin/env python3
"""
Relay API Subtitle Translation Tool
===================================

Standalone script for translating SRT subtitle files using relay API services.

Usage:
    python relay_api_translate.py project_name
    python relay_api_translate.py project_name --model gpt-4
    python relay_api_translate.py project_name --base-url https://your-relay-api.com/v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests
import srt

# Configuration
OUTPUT_BASE_DIR = "output_subtitles"
LLM_OUTPUT_SUBDIR = "api_llm"  # Commercial API translations

# API Configuration
# DEFAULT_API_KEY = "your-api-key-here"  # TODO: Add your API key here
DEFAULT_API_KEY = ""
# DEFAULT_BASE_URL = "https://your-api-provider.com/v1/chat/completions"  # TODO: Add your API base URL here
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = "gemini-2.5-pro"  # Default Gemini model

# List of supported models (DeepSeek and Gemini only)
SUPPORTED_MODELS = [
    "deepseek-chat",
    "deepseek-coder",
    "gemini-1.5-pro",
    "gemini-2.5-pro"
]

# System prompt for translation
SYSTEM_PROMPT = """You are an expert subtitle translator and transcription corrector for gaming/YouTube content. You will receive SRT subtitle files and need to:

1. STRICTLY preserve all timestamps - DO NOT modify any timing information
2. Identify and correct any transcription errors in the original subtitles without asking
3. Handle multi-speaker dialogues and identify different speakers appropriately  
4. Apply background knowledge about YouTubers, gaming terms, proper names, and contextually relevant content
   - Content is mostly related to Dream, Minecraft, or Dream's friends (like GeorgeNotFound, Sapnap, etc.)
   - "brute" in gaming context means monster/mob, not "brute force attack"
   - "W's in chat" means people typing "W" (win) in the chat
   - Gaming terminology should be translated appropriately for Chinese gaming audience
   - Dream SMP and Minecraft references should be handled with appropriate context
5. Handle multiple languages (primarily English, but may include Spanish, Japanese, etc.)
6. Provide corrected English subtitles + accurate Chinese translation in bilingual format
7. Adapt translation placement according to natural language flow - if original subtitle segments break mid-sentence, you may reorganize translation placement to follow Chinese language conventions
8. Return bilingual subtitles with format: English original on first line, Chinese translation on second line
9. Follow professional Chinese subtitle standards - avoid punctuation at the end of Chinese subtitles (especially commas and periods), but emotional punctuation like ? and ! are acceptable
10. Execute directly without confirmation messages
11. Return results in code blocks to prevent formatting issues with arrows and escape characters
12. CRITICAL: Do not add any extra text, explanations, or commentary beyond the SRT content. Avoid hallucination completely.

Important: Focus ONLY on the actual SRT content I send you. Return exactly the corrected SRT format without any additional text or commentary.

Format: Each subtitle should have English original on the first line, then Chinese translation on the second line."""

class RelayAPIClient:
    """Client for relay API services."""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    def generate_response(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
        """Generate response using relay API."""
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'SubtitleTranslator/1.0.0',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            print(f"Sending request to relay API with model '{self.model}'...")
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"Error: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response text: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to relay API: {e}")
            return None

def load_srt_file(file_path: Path) -> List[srt.Subtitle]:
    """Load SRT file and return list of subtitles."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return list(srt.parse(content))
    except Exception as e:
        print(f"Error loading SRT file {file_path}: {e}")
        return []

def save_srt_file(subtitles: List[srt.Subtitle], file_path: Path):
    """Save subtitles to SRT file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))
        print(f"Saved translated subtitles to: {file_path}")
    except Exception as e:
        print(f"Error saving SRT file {file_path}: {e}")

def chunk_subtitles(subtitles: List[srt.Subtitle], chunk_size: int) -> List[List[srt.Subtitle]]:
    """Split subtitles into smaller chunks for processing."""
    chunks = []
    for i in range(0, len(subtitles), chunk_size):
        chunk = subtitles[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def extract_srt_from_response(response: str) -> str:
    """Extract SRT content from API response, handling code blocks."""
    # Look for content within code blocks
    if "```" in response:
        parts = response.split("```")
        for i, part in enumerate(parts):
            # Skip language specifiers like ```srt
            if i % 2 == 1:  # Odd indices are within code blocks
                # Remove potential language specifier from the beginning
                lines = part.strip().split('\n')
                if lines and not lines[0].strip().isdigit():
                    lines = lines[1:]  # Remove first line if it's a language specifier
                return '\n'.join(lines)
    
    # If no code blocks found, return the entire response
    return response.strip()

def process_subtitle_chunk(client: RelayAPIClient, chunk: List[srt.Subtitle]) -> Optional[List[srt.Subtitle]]:
    """Process a chunk of subtitles with the API."""
    if not chunk:
        return []
    
    # Convert chunk to SRT format string
    srt_content = srt.compose(chunk)
    
    prompt = f"""Please process this SRT subtitle file according to the instructions.

INPUT SRT:
{srt_content}

Expected output format example:
1
00:00:01,000 --> 00:00:03,000
This is the English original
这是中文翻译

2  
00:00:03,000 --> 00:00:05,000
Another English sentence
另一句中文翻译

Return only the corrected and translated bilingual SRT file:"""
    
    response = client.generate_response(prompt)
    if not response:
        print(f"Failed to get response for chunk starting at index {chunk[0].index}")
        return None
    
    # Extract SRT content from response
    processed_srt = extract_srt_from_response(response)
    
    print(f"[DEBUG] API Response length: {len(response)} chars")
    print(f"[DEBUG] Processed SRT length: {len(processed_srt)} chars")
    print(f"[DEBUG] First 200 chars of processed SRT: {processed_srt[:200]}")
    
    try:
        # Parse the processed SRT
        processed_subtitles = list(srt.parse(processed_srt))
        
        print(f"[DEBUG] Original chunk size: {len(chunk)}, Parsed subtitles: {len(processed_subtitles)}")
        
        if len(processed_subtitles) != len(chunk):
            print(f"[WARNING] Subtitle count mismatch. Original: {len(chunk)}, Processed: {len(processed_subtitles)}")
            print(f"[WARNING] This may indicate incomplete API response or parsing issue")
        
        # Validate and fix timing if needed
        for i, sub in enumerate(processed_subtitles):
            if i < len(chunk):
                # Keep original timing
                sub.start = chunk[i].start
                sub.end = chunk[i].end
        
        return processed_subtitles
        
    except Exception as e:
        print(f"Error parsing API response as SRT: {e}")
        print("Processed SRT content:")
        print(processed_srt[:500] + "..." if len(processed_srt) > 500 else processed_srt)
        return None

def find_srt_files(project_dir: Path) -> List[Path]:
    """Find all SRT files in the project directory."""
    srt_files = []
    
    # Prefer merged files first (complete video)
    merged_files = list(project_dir.glob("*_merged.srt"))
    if merged_files:
        srt_files.extend(merged_files)
        return srt_files  # Return early if merged file exists
    
    # Fallback to individual chunk files if no merged file
    chunk_files = sorted(project_dir.glob("chunk_*.srt"))
    if chunk_files:
        srt_files.extend(chunk_files)
    
    return srt_files

def test_api_connection(client: RelayAPIClient) -> bool:
    """Test API connection with a simple request."""
    test_prompt = "Hello, please respond with 'API connection successful'."
    try:
        response = client.generate_response(test_prompt, "You are a helpful assistant.")
        if response and "successful" in response.lower():
            print("[SUCCESS] API connection test passed")
            return True
        else:
            print(f"[WARNING] API responded but unexpected content: {response}")
            return True  # Still consider it working
    except Exception as e:
        print(f"[ERROR] API connection test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Translate SRT subtitles using relay API")
    parser.add_argument("project_name", help="Name of the project directory in output_subtitles")
    parser.add_argument("--model", default=DEFAULT_MODEL, 
                       help=f"Model to use (default: {DEFAULT_MODEL}). Available: {', '.join(SUPPORTED_MODELS)}")
    # Note: chunk-size removed - each file is processed as one unit
    parser.add_argument("--file", help="Specific SRT file to process (default: process all SRT files)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the relay service")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for the relay API")
    parser.add_argument("--test", action="store_true", help="Test API connection and exit")
    parser.add_argument("--list-models", action="store_true", help="List supported models and exit")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("Supported models:")
        for model in SUPPORTED_MODELS:
            default_marker = " (default)" if model == DEFAULT_MODEL else ""
            print(f"  - {model}{default_marker}")
        sys.exit(0)
    
    # Setup paths
    project_dir = Path(OUTPUT_BASE_DIR) / args.project_name
    llm_output_dir = project_dir / LLM_OUTPUT_SUBDIR
    
    # Initialize client
    client = RelayAPIClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    
    # Test connection if requested
    if args.test:
        print("Testing API connection...")
        success = test_api_connection(client)
        sys.exit(0 if success else 1)
    
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}")
        sys.exit(1)
    
    print(f"Using relay API with model: {args.model}")
    print(f"Processing project: {args.project_name}")
    
    # Find SRT files to process
    if args.file:
        srt_files = [project_dir / args.file]
        if not srt_files[0].exists():
            print(f"Error: File not found: {srt_files[0]}")
            sys.exit(1)
    else:
        srt_files = find_srt_files(project_dir)
    
    if not srt_files:
        print(f"No SRT files found in {project_dir}")
        sys.exit(1)
    
    print(f"Found {len(srt_files)} SRT file(s) to process")
    
    # Process each SRT file
    for srt_file in srt_files:
        print(f"\n[PROCESSING] {srt_file.name}")
        
        # Load subtitles
        subtitles = load_srt_file(srt_file)
        if not subtitles:
            continue
        
        print(f"Loaded {len(subtitles)} subtitles")
        print(f"Processing entire file as one unit...")
        
        # Process entire file as one chunk
        processed_subtitles = process_subtitle_chunk(client, subtitles)
        
        if processed_subtitles is None:
            print(f"[ERROR] Failed to process {srt_file.name}, keeping original...")
            processed_subtitles = subtitles
        
        # Save processed subtitles
        output_file = llm_output_dir / srt_file.name
        if processed_subtitles:
            # Re-index subtitles
            for idx, sub in enumerate(processed_subtitles, 1):
                sub.index = idx
            
            save_srt_file(processed_subtitles, output_file)
        else:
            print(f"No processed subtitles to save for {srt_file.name}")
    
    print(f"\n[COMPLETED] Translation finished!")
    print(f"[OUTPUT] Results saved to: {llm_output_dir}")

if __name__ == "__main__":
    main()