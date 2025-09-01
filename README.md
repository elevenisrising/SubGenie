# SubGenie - Subtitle Generation & Translation Tool

SubGenie is a desktop application for generating, translating, and refining subtitles for audio and video files. It uses OpenAI's Whisper for transcription and supports various local and API-based LLMs for translation.

## Features

- **Transcription**: Uses OpenAI's Whisper models (tiny to large-v3).
- **Translation**: Supports local LLMs via Ollama, commercial APIs (e.g., DeepSeek, Gemini), and a basic free translation mode.
- **GUI**: A graphical user interface built with CustomTkinter.
- **Output Formats**: Supports SRT, VTT, ASS, TXT, and JSON.
- **Project Management**: Organizes files into project folders.
- **Audio Preprocessing**: Includes options for volume normalization and noise reduction.
- **Glossary**: Allows custom translation rules via `src/utils/glossary.json`.

## Installation Guide

### 1. Prerequisites

- **Python**: Version 3.9 or higher.
- **FFmpeg**: Required for audio processing.
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system's PATH, or install via a package manager like Chocolatey (`choco install ffmpeg`).
  - **macOS**: Install using Homebrew: `brew install ffmpeg`.
  - **Linux (Debian/Ubuntu)**: Install using apt: `sudo apt install ffmpeg`.
- **Ollama** (Optional): Required only for using local LLMs for translation. Install from [ollama.com](https://ollama.com/).

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/elevenisrising/SubGenie.git
    cd SubGenie
    ```

2.  **Create and activate a Python environment:**
    - Using **conda** (recommended):
      ```bash
      conda create -n subgenie python=3.10 -y
      conda activate subgenie
      ```
    - Using **venv**:
      ```bash
      python -m venv .venv
      # On Windows: .venv\Scripts\activate
      # On macOS/Linux: source .venv/bin/activate
      ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the NLP model:**
    This is required for sentence segmentation.
    ```bash
    python -m spacy download en_core_web_sm
    ```

### 3. Launching the Application

-   **Windows**: Double-click `start_gui.bat`.
-   **macOS/Linux**: Run `python main_gui.py` in your terminal.

## Usage Manual

### Basic Workflow

1.  **Add Files**: Drag and drop files onto the application or use the "Add Files" / "Add Folder" buttons.
2.  **Configure**:
    -   **Whisper Model**: Select the model size (e.g., `medium`, `large`). Larger models are more accurate but slower.
    -   **Language**: Set the source language or leave as auto-detect.
    -   **Output**: Choose the output format (e.g., `source`, `bilingual`).
3.  **Process**: Click "Start Processing". Progress and logs will be displayed in the text area at the bottom.
4.  **Output**: The generated subtitle files will be located in the `output_subtitles/` directory, organized by project name.

### Advanced Usage

-   **LLM Translation**:
    -   **Local**: Select "Local LLM" in the "Advanced Options" tab. Requires Ollama to be running with a downloaded model.
    -   **API**: Select "API LLM" and enter your API key, base URL, and model name.
-   **Merge Subtitles**: Use the "Merge Subtitles" tab to combine `chunk_*.srt` files from a project folder into a single merged subtitle file.

## Troubleshooting

-   **`ffmpeg not found`**: Ensure FFmpeg is installed and its location is included in your system's PATH environment variable.
-   **`spaCy model 'en_core_web_sm' not found`**: Run `python -m spacy download en_core_web_sm` again in your activated environment.
-   **CUDA Errors with PyTorch/Whisper**: If you encounter GPU-related errors, you may need to install a specific version of PyTorch that matches your CUDA toolkit, or install the CPU-only version.
-   **Permissions**: On macOS, you may need to grant terminal or the application access to your files.

## Project Structure

```
SubGenie/
├── core/              # Core processing logic
├── gui/               # User interface components
├── src/               # Main source code
├── input_audio/       # Default directory for input files
├── output_subtitles/  # Default directory for output files
├── main_gui.py        # Main application entry point
├── start_gui.bat      # Windows launcher script
└── requirements.txt   # Python dependencies
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.