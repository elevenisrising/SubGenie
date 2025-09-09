#!/usr/bin/env python3
"""
LLM-based SRT Translation Tool using Ollama
==========================================

This tool processes SRT subtitle files using local LLM models via Ollama
to correct transcription errors and add Chinese translations.

Usage:
    # Use local Ollama (free)
    python llm_translate.py project_name
    python llm_translate.py project_name --model qwen2.5:7b
    
    # Use commercial APIs (paid)
    python llm_translate.py project_name --api openai --model gpt-4o-mini
    python llm_translate.py project_name --api anthropic --model claude-3-haiku-20240307
    python llm_translate.py project_name --api deepseek --api-key your_key_here
    python llm_translate.py project_name --api zhipuai
    
    # Environment variables for API keys
    export OPENAI_API_KEY="your_openai_key"
    export ANTHROPIC_API_KEY="your_anthropic_key" 
    export DEEPSEEK_API_KEY="your_deepseek_key"
    export ZHIPUAI_API_KEY="your_zhipuai_key"
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

# Configuration - use dynamic paths
try:
    from src.utils.constants import get_output_dir, OUTPUT_DIR_NAME
    OUTPUT_BASE_DIR = OUTPUT_DIR_NAME  # For backward compatibility
except ImportError:
    OUTPUT_BASE_DIR = "output_subtitles"
LLM_OUTPUT_SUBDIR = "local_llm"  # Local LLM translations
DEFAULT_MODEL = "qwen2.5:7b"  # Recommended for Chinese translation
OLLAMA_BASE_URL = "http://localhost:11434"

# API Configuration
SUPPORTED_API_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini"
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "default_model": "claude-3-5-sonnet-20241022"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "models": ["deepseek-chat", "deepseek-coder"],
        "default_model": "deepseek-chat"
    },
    "zhipuai": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "models": ["glm-4-flash", "glm-4"],
        "default_model": "glm-4-flash"
    }
}

# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert subtitle translator and transcription corrector for gaming/YouTube content. You will receive SRT subtitle files and need to:

1. STRICTLY preserve all timestamps - DO NOT modify any timing information
2. Identify and correct any transcription errors in the original subtitles without asking
3. Handle multi-speaker dialogues and identify different speakers appropriately  
4. Apply background knowledge about YouTubers, gaming terms, proper names, and contextually relevant content
   - "brute" in gaming context means monster/mob, not "brute force attack"
   - "W's in chat" means people typing "W" (win) in the chat
   - Gaming terminology should be translated appropriately for Chinese gaming audience
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

class APIClient:
    """Client for interacting with commercial API providers."""
    
    def __init__(self, provider: str, model: str = None, api_key: str = None):
        if provider not in SUPPORTED_API_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(SUPPORTED_API_PROVIDERS.keys())}")
        
        self.provider = provider
        self.config = SUPPORTED_API_PROVIDERS[provider]
        self.model = model or self.config["default_model"]
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        if not self.api_key:
            raise ValueError(f"API key required for {provider}. Set {provider.upper()}_API_KEY environment variable or pass --api-key")
    
    def generate_response(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
        """Generate response from API provider."""
        if self.provider == "anthropic":
            return self._call_anthropic_api(prompt, system_prompt)
        else:
            return self._call_openai_compatible_api(prompt, system_prompt)
    
    def _call_openai_compatible_api(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Call OpenAI-compatible API (OpenAI, DeepSeek, ZhipuAI)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            print(f"Sending request to {self.provider} API with model '{self.model}'...")
            response = requests.post(self.config["base_url"], headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"Error: HTTP {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to {self.provider} API: {e}")
            return None
    
    def _call_anthropic_api(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Call Anthropic Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            print(f"Sending request to Anthropic API with model '{self.model}'...")
            response = requests.post(self.config["base_url"], headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"].strip()
            else:
                print(f"Error: HTTP {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to Anthropic API: {e}")
            return None

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    def check_connection(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def check_model_available(self) -> bool:
        """Check if the specified model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                return any(self.model in model_name for model_name in available_models)
            return False
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
        """Generate response from Ollama model."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent output
                "top_p": 0.8,
                "max_tokens": 4000,
                "repeat_penalty": 1.1,  # Reduce repetition
                "stop": ["---", "Note:", "注意:", "解释:", "Explanation:"]  # Stop tokens to prevent explanations
            }
        }
        
        try:
            print(f"Sending request to LLM model '{self.model}'...")
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Error: HTTP {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Error: Request timed out. The model might be too large or the server is overloaded.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to Ollama: {e}")
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
    """Extract SRT content from LLM response, handling code blocks and cleaning up."""
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
                return clean_srt_content('\n'.join(lines))
    
    # If no code blocks found, return the entire response
    return clean_srt_content(response.strip())

def clean_srt_content(content: str) -> str:
    """Clean up SRT content to remove hallucinations and extra text."""
    lines = content.split('\n')
    cleaned_lines = []
    in_srt_block = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines at the beginning
        if not line and not in_srt_block:
            continue
            
        # Check if line looks like subtitle index (just a number)
        if line.isdigit():
            in_srt_block = True
            cleaned_lines.append(line)
            continue
            
        # Check if line looks like timestamp
        if '-->' in line and ':' in line:
            in_srt_block = True
            cleaned_lines.append(line)
            continue
            
        # If we're in SRT block and it's subtitle content
        if in_srt_block:
            # Keep the line if it looks like subtitle content
            cleaned_lines.append(line)
        
        # Add empty line after subtitle content
        if line == '' and cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def process_subtitle_chunk(client: OllamaClient, chunk: List[srt.Subtitle]) -> Optional[List[srt.Subtitle]]:
    """Process a chunk of subtitles with the LLM."""
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
    
    try:
        # Parse the processed SRT
        processed_subtitles = list(srt.parse(processed_srt))
        
        if len(processed_subtitles) != len(chunk):
            print(f"Warning: Subtitle count mismatch. Original: {len(chunk)}, Processed: {len(processed_subtitles)}")
        
        # Validate bilingual format
        validated_subtitles = validate_bilingual_format(processed_subtitles, chunk)
        
        return validated_subtitles
        
    except Exception as e:
        print(f"Error parsing LLM response as SRT: {e}")
        print("Cleaned SRT content:")
        print(processed_srt[:500] + "..." if len(processed_srt) > 500 else processed_srt)
        return None

def validate_bilingual_format(processed_subs: List[srt.Subtitle], original_chunk: List[srt.Subtitle]) -> List[srt.Subtitle]:
    """Validate and fix bilingual subtitle format."""
    validated = []
    
    for i, sub in enumerate(processed_subs):
        # Ensure we have the original timing if something went wrong
        if i < len(original_chunk):
            # Keep original timing
            sub.start = original_chunk[i].start
            sub.end = original_chunk[i].end
        
        # Check if content has both Chinese and English
        lines = sub.content.strip().split('\n')
        
        # If only one line, assume it's Chinese and we need to add English
        if len(lines) == 1:
            chinese_line = lines[0]
            english_line = original_chunk[i].content if i < len(original_chunk) else ""
            sub.content = f"{chinese_line}\n{english_line}"
        # If multiple lines, keep as is but clean up
        else:
            # Clean up any remaining artifacts
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            if len(cleaned_lines) >= 2:
                sub.content = '\n'.join(cleaned_lines[:2])  # Take first 2 lines only
            elif len(cleaned_lines) == 1:
                # Add original English if only Chinese exists
                english_line = original_chunk[i].content if i < len(original_chunk) else ""
                sub.content = f"{cleaned_lines[0]}\n{english_line}"
        
        validated.append(sub)
    
    return validated

def find_srt_files(project_dir: Path) -> List[Path]:
    """Find all SRT files in the project directory."""
    srt_files = []
    
    # Look for individual chunk files
    chunk_files = sorted(project_dir.glob("chunk_*.srt"))
    if chunk_files:
        srt_files.extend(chunk_files)
    
    # Look for merged file
    merged_files = list(project_dir.glob("*_merged.srt"))
    if merged_files:
        srt_files.extend(merged_files)
    
    return srt_files

def main():
    parser = argparse.ArgumentParser(description="Translate SRT subtitles using local LLM via Ollama or commercial APIs")
    parser.add_argument("project_name", help="Name of the project directory in output_subtitles")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL} for Ollama)")
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of subtitles to process in each batch (default: 5)")
    parser.add_argument("--file", help="Specific SRT file to process (default: process all SRT files)")
    
    # API options
    parser.add_argument("--api", choices=list(SUPPORTED_API_PROVIDERS.keys()), 
                       help="Use commercial API instead of Ollama (openai, anthropic, deepseek, zhipuai)")
    parser.add_argument("--api-key", help="API key for commercial provider (or set environment variable)")
    
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(OUTPUT_BASE_DIR) / args.project_name
    llm_output_dir = project_dir / LLM_OUTPUT_SUBDIR
    
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}")
        sys.exit(1)
    
    # Initialize client (API or Ollama)
    if args.api:
        try:
            # Use API model if specified, otherwise use provider's default
            api_model = args.model if args.model != DEFAULT_MODEL else None
            client = APIClient(provider=args.api, model=api_model, api_key=args.api_key)
            print(f"Using {args.api.upper()} API with model: {client.model}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Use Ollama
        client = OllamaClient(model=args.model)
        
        # Check Ollama connection
        if not client.check_connection():
            print(f"Error: Cannot connect to Ollama server at {OLLAMA_BASE_URL}")
            print("Please make sure Ollama is running: ollama serve")
            sys.exit(1)
        
        # Check if model is available
        if not client.check_model_available():
            print(f"Error: Model '{args.model}' is not available")
            print(f"Please pull the model first: ollama pull {args.model}")
            sys.exit(1)
        
        print(f"Using Ollama with model: {args.model}")
    
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
        print(f"\nProcessing: {srt_file.name}")
        
        # Load subtitles
        subtitles = load_srt_file(srt_file)
        if not subtitles:
            continue
        
        print(f"Loaded {len(subtitles)} subtitles")
        
        # Split into chunks for processing
        chunks = chunk_subtitles(subtitles, args.chunk_size)
        print(f"Processing in {len(chunks)} chunk(s)")
        
        processed_subtitles = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            
            processed_chunk = process_subtitle_chunk(client, chunk)
            if processed_chunk is not None:
                processed_subtitles.extend(processed_chunk)
            else:
                print(f"Failed to process chunk {i}, skipping...")
                # Keep original chunk if processing fails
                processed_subtitles.extend(chunk)
            
            # Add small delay to avoid overwhelming the model
            if i < len(chunks):
                time.sleep(1)
        
        # Save processed subtitles
        output_file = llm_output_dir / srt_file.name
        if processed_subtitles:
            # Re-index subtitles
            for idx, sub in enumerate(processed_subtitles, 1):
                sub.index = idx
            
            save_srt_file(processed_subtitles, output_file)
        else:
            print(f"No processed subtitles to save for {srt_file.name}")
    
    print(f"\n[COMPLETED] LLM translation finished!")
    print(f"Results saved to: {llm_output_dir}")

if __name__ == "__main__":
    main()