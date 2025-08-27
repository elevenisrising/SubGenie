# Whisper Subtitle Generator

An advanced, all-in-one tool to automatically generate and translate subtitles from any audio or video file using Whisper, featuring intelligent audio chunking, preprocessing, and glossary support.

---

## Features

- **High-Quality Transcription**: Utilizes OpenAI's Whisper models for accurate speech-to-text conversion.
- **Multi-Language Translation**: Integrated with Google Translate to provide bilingual or target-language-only subtitles.
- **Intelligent Audio Chunking**: Smartly splits long audio files into manageable chunks, avoiding cuts in the middle of speech.
- **Audio Preprocessing**: Includes options for volume normalization and noise reduction to improve transcription accuracy.
- **Glossary Support**: Customize translations for specific terms, names, or jargon using a simple JSON file.
- **Multiple Output Formats**: Generate subtitles in `.srt`, `.vtt`, `.ass`, `.txt`, and `.json` formats.
- **Performance Caching**: Caches transcription results to dramatically speed up subsequent runs on the same audio.
- **Highly Configurable**: Provides a wide range of command-line arguments to tune performance, quality, and output.

## Project Structure

```
.
├── input_audio/         # Place your source audio/video files here.
├── output_subtitles/    # Generated subtitles and audio chunks are saved here.
├── .cache/              # Caches transcription results to speed up re-runs.
├── __pycache__/         # Directory for Python's compiled bytecode.
├── .gitignore           # Specifies files and directories to be ignored by Git.
├── main.py              # The main script for generating subtitles.
├── requirements.txt     # A list of Python dependencies for the project.
├── download_nltk_data.py  # Standalone script to pre-download NLTK data.
├── extract_audio.py     # A utility to extract audio from video files.
├── merge_subtitles.py   # A utility to merge subtitle chunks manually.
└── README.md            # This documentation file.
```

### Script Descriptions

- **`main.py`**: The primary entry point of the application. It handles argument parsing, audio chunking, preprocessing, transcription, translation, and subtitle generation.
- **`download_nltk_data.py`**: A helper script to download the necessary NLTK datasets. `main.py` calls this automatically, but you can run it manually to set up the environment.
- **`extract_audio.py`**: A utility script to extract audio tracks from video files before processing. While `main.py` can handle video directly, extracting audio first can be more efficient.
- **`merge_subtitles.py`**: A helper script to manually merge individual subtitle chunk files (`.srt`) into a single file.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **Pip** (Python package installer)
- **FFmpeg**: This is required for audio extraction from video files.
  - **Windows**: `choco install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install ffmpeg`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. NLTK Data (Automatic Download)

The first time you run `main.py`, it will automatically check for and download the necessary NLTK `punkt_tab` dataset if it's not found. **This is a one-time setup.** You do not need to run anything manually.

For troubleshooting or manual setup, you can run the download script directly, but this is optional:
```bash
python download_nltk_data.py
```

## Usage

1.  Place your audio or video files into the `input_audio` directory.
2.  Run the main script from your terminal, pointing to your desired file.

### Quick Start

To generate bilingual (source + Chinese) subtitles for a file using the default settings:
```bash
python main.py my_audio.mp4
```

The output will be saved in the `output_subtitles/my_audio/` directory.

### Command-Line Arguments

- `filename`: (Required) The name of the input audio or video file.
- `--model`: The Whisper model to use. (Default: `medium`)
- `--target_language`: Target language code for translation. (Default: `zh-CN`)
- `--output_format`: Subtitle format. Options: `bilingual`, `source`, `target`. (Default: `bilingual`)
- ... (For a full list of arguments, see the script's help message with `python main.py -h`) ...

## Glossary Feature

You can create a `glossary.json` file in your project's output directory (`output_subtitles/<project_name>/`) to enforce specific translation rules.

**Example `glossary.json`:**
```json
{
  "exact_replace": {
    "Minecraft": "我的世界",
    "Whisper": "Whisper模型"
  },
  "pre_translate": {
    "Hello World": "你好，世界！"
  }
}
```

## License

This project is licensed under the MIT License.