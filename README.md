# SubGenie - AI-Powered Subtitle Generator

**SubGenie** is a desktop application designed for content creators to generate, translate, and refine subtitles from audio/video files using state-of-the-art AI models. It specializes in gaming content with extensive terminology support for Minecraft, speedrunning, and streaming contexts.

## 🚀 Features

### Core Functionality
- **Advanced ASR**: Uses OpenAI Whisper with WhisperX forced alignment for word-level timestamps
- **Smart Segmentation**: Grammar-aware sentence splitting using spaCy NLP with 5-tier priority system
- **Multi-format Output**: SRT, VTT, ASS, TXT, JSON subtitle formats
- **Bilingual Subtitles**: Automatic translation with gaming-specific terminology
- **Chunk Processing**: Handles long audio files (>15min) with automatic chunking

### Translation System
- **Local LLM Support**: Ollama integration for free, offline translation
- **Commercial APIs**: OpenAI, Anthropic, Google Gemini, DeepSeek support
- **Gaming Glossary**: 700+ term dictionary for gaming/streaming content
- **Context-Aware**: Specialized prompts for gaming, speedrunning, and creator content

### Gaming Content Specialization  
- **Creator Names**: Proper handling of content creator identities
- **Minecraft Terminology**: Blocks, items, mechanics, speedrunning terms
- **Streaming Context**: Chat interactions, donations, subscriber content
- **Community Terms**: Dream SMP, popular gaming phrases and slang

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **FFmpeg** (system dependency for audio processing)
- **Conda/Anaconda** (recommended for environment management)

### Required Python Packages
```bash
# Core dependencies
pip install openai-whisper whisperx
pip install spacy && python -m spacy download en_core_web_sm
pip install customtkinter pillow
pip install pydub torch torchaudio
pip install srt deep-translator requests
pip install numpy tqdm

# Optional for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd SubGenie

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Launch GUI
python main_gui.py
```

## 🎯 Usage

### GUI Application (Recommended)
```bash
# Windows (with Conda)
start_gui.bat

# Cross-platform
python main_gui.py
python subgenie.py  # Alternative entry point
```

### CLI Processing
```bash
# Basic transcription
python src/processing/main.py input.mp4 --model medium

# With translation
python src/processing/main.py input.mp4 --model large-v3 --target-language zh-CN

# Custom settings
python src/processing/main.py input.mp4 --max_subtitle_chars 60 --chunk_duration 20
```

## Usage Manual

### Basic Workflow

1.  **Add Files**: Drag and drop files onto the application or use the "Add Files" / "Add Folder" buttons.
2.  **Configure**:
    -   **Whisper Model**: Select the model size (e.g., `medium`, `large`). Larger models are more accurate but slower.
    -   **Language**: Set the source language or leave as auto-detect. Target language defaults to "none" (no translation).
    -   **Segmentation**: Choose between "spaCy Grammar" (recommended) or "Whisper Segments".
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

---

# SubGenie - 字幕生成与翻译工具 (简体中文)

SubGenie 是一款桌面应用程序，用于生成、翻译和优化音频和视频文件的字幕。它使用 OpenAI 的 Whisper进行语音转录，并支持通过Ollama调用本地大语言模型（LLM）或调用商业API（如DeepSeek、Gemini）进行翻译。

## 功能特性

- **语音转录**: 使用OpenAI的Whisper模型（从`tiny`到`large-v3`）进行词级时间戳转录。
- **智能分句**: 基于spaCy语法的高级分句技术，生成自然的句子边界。
- **字幕翻译**: 支持通过Ollama运行的本地大语言模型、商业API（例如DeepSeek、Gemini）以及免费的谷歌翻译。
- **图形界面**: 基于CustomTkinter构建的图形用户界面，支持实时处理日志。
- **输出格式**: 支持SRT、VTT、ASS、TXT和JSON格式。
- **项目管理**: 将每个任务的文件整理到单独的项目文件夹中，支持分块处理。
- **音频预处理**: 提供音量标准化和降噪等选项。
- **精确时间戳匹配**: 使用滑动窗口算法实现精确的词句时间戳对齐。
- **灵活处理**: 支持并行分块处理，具备错误恢复和重处理能力。
- **术语表**: 允许通过`src/utils/glossary.json`文件自定义翻译规则。

## 安装指南

### 1. 系统要求

- **Python**: 3.9或更高版本。
- **FFmpeg**: 音频处理所必需的工具。
  - **Windows**: 从[ffmpeg.org](https://ffmpeg.org/download.html)下载并添加到系统PATH环境变量，或通过包管理器（如Chocolatey）安装：`choco install ffmpeg`。
  - **macOS**: 使用Homebrew安装：`brew install ffmpeg`。
  - **Linux (Debian/Ubuntu)**: 使用apt安装：`sudo apt install ffmpeg`。
- **Ollama** (可选): 仅在使用本地大语言模型进行翻译时需要。从[ollama.com](https://ollama.com/)安装。

### 2. 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/elevenisrising/SubGenie.git
    cd SubGenie
    ```

2.  **创建并激活Python环境:**
    - **conda** (推荐):
      ```bash
      conda create -n subgenie python=3.10 -y
      conda activate subgenie
      ```
    - **venv**:
      ```bash
      python -m venv .venv
      # Windows系统: .venv\Scripts\activate
      # macOS/Linux系统: source .venv/bin/activate
      ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **下载spaCy NLP模型:**
    此模型为高级语法分句所必需。
    ```bash
    python -m spacy download en_core_web_sm
    ```

### 3. 启动应用

-   **Windows**: 双击`start_gui.bat`。
-   **macOS/Linux**: 在终端中运行`python main_gui.py`。

## 使用手册

### 基本流程

1.  **添加文件**: 将文件拖放到应用程序中，或使用“添加文件”/“添加文件夹”按钮。
2.  **配置**:
    -   **Whisper模型**: 选择模型大小（例如`medium`, `large`）。模型越大，准确率越高，但速度越慢。
    -   **语言**: 设置源语言，或保留为自动检测。目标语言默认为"none"（不翻译）。
    -   **分句策略**: 选择"spaCy Grammar"（推荐）或"Whisper Segments"。
    -   **输出**: 选择输出格式（例如`source` - 仅源语言, `bilingual` - 双语）。
3.  **处理**: 点击“开始处理”。处理进度和日志将显示在底部的文本区域。
4.  **输出**: 生成的字幕文件将位于`output_subtitles/`目录中，按项目名称分类。

### 高级用法

-   **LLM翻译**:
    -   **本地**: 在“高级选项”选项卡中选择“本地LLM”。需要Ollama正在运行并且已下载相应模型。
    -   **API**: 选择“API LLM”，并输入您的API密钥、基础URL和模型名称。
-   **合并字幕**: 使用“合并字幕”选项卡，将项目文件夹中的`chunk_*.srt`文件合并为单个完整的字幕文件。

## 常见问题

-   **`ffmpeg not found`**: 确保FFmpeg已安装，并且其路径已添加到系统的PATH环境变量中。
-   **`spaCy model 'en_core_web_sm' not found`**: 在激活的Python环境中再次运行`python -m spacy download en_core_web_sm`。
-   **PyTorch/Whisper的CUDA错误**: 如果遇到与GPU相关的错误，您可能需要安装与您的CUDA工具包匹配的特定版本的PyTorch，或安装仅CPU版本。
-   **权限问题**: 在macOS上，您可能需要授予终端或应用程序访问文件的权限。

## 项目结构

```
SubGenie/
├── core/              # 核心处理逻辑
├── gui/               # 用户界面组件
├── src/               # 主要源代码
├── input_audio/       # 输入文件的默认目录
├── output_subtitles/  # 输出文件的默认目录
├── main_gui.py        # 应用程序主入口
├── start_gui.bat      # Windows启动脚本
└── requirements.txt   # Python依赖
```

## 许可证

本项目基于MIT许可证。详情请参阅[LICENSE](LICENSE)文件。
