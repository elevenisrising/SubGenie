# Advanced Subtitle Generation & Translation Tool

An all-in-one tool for generating, processing, and translating subtitles from audio/video files using OpenAI Whisper, spaCy, and various LLM APIs.

---

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper models with spaCy for intelligent sentence segmentation
- **LLM-Powered Translation**: Support for local Ollama and commercial APIs (DeepSeek, Gemini) 
- **Intelligent Audio Chunking**: Smart splitting avoiding speech interruption
- **Audio Preprocessing**: Volume normalization and noise reduction options
- **Glossary Support**: Custom translation rules via JSON configuration
- **Multiple Output Formats**: SRT, VTT, ASS, TXT, and JSON formats
- **Chunk Reprocessing**: Selectively reprocess specific audio segments
- **Relay API Support**: Cost-effective API access through relay services

## Project Structure

```
.
├── input_audio/          # Source audio/video files
├── output_subtitles/     # Generated subtitles and chunks
├── V1/                   # Legacy NLTK-based versions
│   ├── main.py
│   ├── reprocess_chunk.py
│   ├── extract_audio.py
│   └── download_nltk_data.py
├── main.py               # Main spaCy-based subtitle generator
├── reprocess_chunk.py    # Chunk reprocessing tool
├── llm_translate.py      # LLM translation (Ollama + APIs)
├── relay_api_translate.py # Relay API translation tool
├── merge_subtitles.py    # Manual subtitle merging utility
├── glossary.json         # Global translation rules
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites
- Python 3.9+
- FFmpeg for audio/video processing
  - Windows: `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd translate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Subtitle Generation

**Generate bilingual subtitles:**
```bash
python main.py audio_file.mp4
```

**Use different Whisper model:**
```bash
python main.py audio_file.mp4 --model large
```

### LLM Translation

**Using Ollama (local, free):**
```bash
# Setup Ollama first: https://ollama.ai
ollama serve
ollama pull qwen2.5:7b

python llm_translate.py project_name
```

**Using Relay API (paid, higher quality):**
```bash
python relay_api_translate.py project_name
python relay_api_translate.py project_name --model deepseek-chat
```

### Chunk Reprocessing

**Reprocess specific chunks:**
```bash
python reprocess_chunk.py project_name 1,3,5 --model large
python reprocess_chunk.py project_name all --model large
```

## Configuration

### Glossary System

Create `glossary.json` in project output directory:

```json
{
  "exact_replace": {
    "Dream": "Dream",
    "GeorgeNotFound": "乔治",
    "Sapnap": "Sapnap",
    "Minecraft": "我的世界"
  },
  "pre_translate": {
    "What's up chat": "聊天室的朋友们好"
  }
}
```

### API Keys (for LLM translation)

Set environment variables or use command-line arguments:

```bash
# Environment variables
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"

# Or use --api-key argument
python llm_translate.py project_name --api openai --api-key your_key
```

## Command Reference

### main.py
```bash
python main.py filename [options]
  --model           Whisper model (default: medium)
  --target_language Translation target (default: zh-CN)
  --output_format   bilingual|source|target (default: bilingual)
  --max_subtitle_chars  Max chars per line (default: 80)
```

### reprocess_chunk.py
```bash
python reprocess_chunk.py project_name chunks [options]
  chunks            1, 1,2,3, 1-3, or "all"
  --model          Whisper model (default: large)
  --use-preprocessed Use denoised audio if available
```

### llm_translate.py
```bash
python llm_translate.py project_name [options]
  --model          Model name (default: qwen2.5:7b)
  --api            openai|anthropic|deepseek|zhipuai
  --chunk-size     Subtitles per batch (default: 5)
```

### relay_api_translate.py
```bash
python relay_api_translate.py project_name [options]
  --model          deepseek-chat|gemini-2.5-pro (default: gemini-2.5-pro)
  --chunk-size     Subtitles per batch (default: 3)
  --test           Test API connection
```

## Workflow Examples

### Complete Processing Pipeline
```bash
# 1. Generate initial subtitles
python main.py dream_video.mp4 --model large

# 2. Improve with LLM translation
python relay_api_translate.py dream_video --model gemini-2.5-pro

# 3. Reprocess problematic chunks
python reprocess_chunk.py dream_video 3,7,12 --model large
```

### Cost-Effective Translation
```bash
# Use cheaper DeepSeek model with larger batches
python relay_api_translate.py project_name --model deepseek-chat --chunk-size 10
```

## Supported Models

### Whisper Models
- `tiny`, `base`, `small`, `medium`, `large`

### LLM Models (via APIs)
- **DeepSeek**: `deepseek-chat`, `deepseek-coder`
- **Gemini**: `gemini-1.5-pro`, `gemini-2.5-pro`

### Local LLM (via Ollama)
- `qwen2.5:7b`, `llama2`, `codellama`, etc.

## Troubleshooting

**spaCy model missing:**
```bash
python -m spacy download en_core_web_sm
```

**Ollama connection error:**
```bash
ollama serve
# In another terminal:
ollama pull qwen2.5:7b
```

**API rate limits:**
- Increase `--chunk-size` to reduce requests
- Add delays between requests
- Use cheaper models like `deepseek-chat`

## License

MIT License - See LICENSE file for details.