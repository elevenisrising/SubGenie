# SubGenie - AI-Powered Subtitle Generator

SubGenie delivers end-to-end subtitle generation, translation, and polishing for creators. It specialises in gaming and livestream content, providing word-level timestamps with WhisperX alignment and spaCy-guided sentence segmentation.

## Features
- **Word-level ASR**: GPU-accelerated WhisperX (via PyTorch CUDA 12.1) with forced alignment for per-word timestamps.
- **Grammar-aware segmentation**: spaCy (`en_core_web_sm`) splits aligned words into natural sentences before subtitle packaging.
- **Gaming-focused translation**: Local LLM (Ollama) or API providers (OpenAI, Anthropic, Gemini, DeepSeek) enriched with a 700+ term glossary.
- **Multi-format output**: Generates SRT/VTT/ASS/TXT/JSON with bilingual presets, chunked processing, and reprocessing utilities.
- **GUI & CLI**: CustomTkinter desktop interface plus a flexible CLI pipeline for batch automation.

## Requirements
- Windows 10/11 (tested) with NVIDIA GPU for best performance (CUDA 12.1 runtime included in the environment).
- [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html) for managing the `asr-env` prefix.
- Adequate disk space for caching ASR/translation models and Hugging Face downloads.

> The project ships with a fully pinned environment under `environment/conda-env.yml`. FFmpeg 6.1.1, CUDA 12.9 components, cuDNN 8.9.7, and all Python dependencies are locked to the versions currently in use.

## Quick Start (Conda prefix)
```powershell
# 1. Create the exact environment prefix used by the GUI launcher
d:
cd D:\Dream\translate
conda env create -p .conda\asr-env -f environment\conda-env.yml

# 2. (Optional) Refresh pip wheels inside the prefix
conda run -p .conda\asr-env python -m pip install -r requirements.txt

# 3. Launch the GUI (handles PATH/FFmpeg automatically)
start_gui.bat
```

`start_gui.bat` prefers the path-based prefix (`D:\Dream\translate\.conda\asr-env`). It injects `Library\bin` to the front of `PATH` so the bundled FFmpeg/CUDA DLLs are used, then runs `main_gui.py` via `conda run -p ...`.

### CLI Examples
```powershell
# Transcribe with WhisperX + spaCy segmentation + word timestamps
conda run -p .conda\asr-env python src/processing/main.py ^
  input_audio\100k.mp4 ^
  --model medium ^
  --output_format source ^
  --segmentation_strategy spacy ^
  --word_timestamps

# Transcribe + translate to Simplified Chinese, using bilingual output
d:\Dream\translate\.conda\asr-env\python.exe -u src/processing/main.py ^
  input_audio\example.mp4 ^
  --model large-v3 ^
  --target_language zh-CN ^
  --output_format bilingual ^
  --word_timestamps
```
(Using `--word_timestamps` ensures the post-processing pipeline receives per-word alignment, which spaCy then groups into sentence-level subtitles.)

## Processing Flow (Overview)
1. **Extraction** - FFmpeg (from the Conda prefix) converts media to WAV per chunk.
2. **ASR** - WhisperX loads the specified Whisper model (`medium` by default) on CUDA 12.1.
3. **Alignment** - WhisperX alignment produces word-level timestamps.
4. **Segmentation** - spaCy splits the transcript into sentences; fallback regex is used only if spaCy fails to load.
5. **Packaging** - Subtitles are assembled respecting `--max_subtitle_chars`, output format, and optional translation mode.

## Updating or Sharing the Environment
- Re-export after upgrades:
  ```powershell
  conda env export -p D:\Dream\translate\.conda\asr-env --no-builds > environment\conda-env.yml
  conda run -p D:\Dream\translate\.conda\asr-env python -m pip freeze > requirements.txt
  ```
- To reuse elsewhere, copy `environment/conda-env.yml` (preferred) and recreate with `conda env create -p <new_prefix> -f ...`. Copying the entire `.conda\asr-env` directory also works on matching Windows/CUDA driver setups but is less portable.

## Troubleshooting
- **spaCy model missing** - run `conda run -p .conda\asr-env python -m spacy download en_core_web_sm`.
- **GPU unavailable** - ensure NVIDIA drivers meet CUDA 12.1 requirements or switch to CPU by reinstalling CPU-only PyTorch.
- **FFmpeg errors** - confirm you are launching through `start_gui.bat` so the prefix DLLs take precedence.
- **Pyannote warnings** - the shipped diarisation model targets older torch/pyannote builds; warnings are benign unless alignment quality degrades.

## Project Structure
```
assets/               # Localisation resources (e.g. en.json, zh.json)
core/                 # GUI orchestration (audio extraction, processor)
environment/          # Environment lockfiles (conda-env.yml)
gui/                  # CustomTkinter interface modules
input_audio/          # Default source directory
output_subtitles/     # Generated subtitles (per run)
scripts/              # Helper batch scripts (create env, CLI wrappers)
src/                  # Main processing pipeline (ASR, segmentation, translation)
start_gui.bat         # Windows launcher selecting .conda\asr-env
```

## License
See [LICENSE](LICENSE) for details.
