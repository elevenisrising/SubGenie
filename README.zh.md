[English](README.md) | [中文](README.zh.md)

# SubGenie - AI 字幕生成工具

SubGenie 为内容创作者提供端到端的字幕生成、翻译和润色能力。项目针对游戏与直播场景进行了大量优化，结合 WhisperX 词级时间戳与 spaCy 句法分句，能够生成高质量字幕。

## 功能特点
- **词级识别**：基于 GPU 的 WhisperX（PyTorch CUDA 12.1），并通过强制对齐获得每个词的起止时间。
- **语法分句**：默认使用 spaCy (`en_core_web_sm`) 对词序列进行句法切分，失败时自动回退正则规则。
- **游戏化翻译**：支持本地 LLM（Ollama）与多家商业 API，同时内置 700+ 游戏/直播术语词表。
- **多格式输出**：可导出 SRT/VTT/ASS/TXT/JSON 等多种格式，并支持双语字幕、分块重试。
- **GUI + CLI**：提供 CustomTkinter 桌面界面与可脚本化的命令行流程。

## 环境要求
- Windows 10/11（已验证）并搭配 NVIDIA GPU；环境快照包含 CUDA 12.1 运行时。
- 安装 [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html) 以管理 `asr-env` 前缀。
- 预留足够磁盘空间，用于缓存 ASR/翻译模型以及 Hugging Face 下载内容。

> 完整依赖锁定在 `environment/conda-env.yml`，其中包含 FFmpeg 6.1.1、CUDA 12.9 组件、cuDNN 8.9.7 以及所有 Python 包版本。

## Conda 前缀快速创建
```powershell
# 1. 在仓库根目录创建与 GUI 启动脚本一致的前缀
d:
cd D:\Dream\translate
conda env create -p .conda\asr-env -f environment\conda-env.yml

# 2. （可选）刷新 pip 依赖
conda run -p .conda\asr-env python -m pip install -r requirements.txt

# 3. 启动 GUI（自动处理 PATH 与 FFmpeg）
start_gui.bat
```

`start_gui.bat` 会优先检测 `D:\Dream\translate\.conda\asr-env`，将该前缀的 `Library\bin` 插入 `PATH` 前端以使用本地 FFmpeg/CUDA DLL，然后执行 `conda run -p ... python main_gui.py` 启动界面。

### CLI 使用示例
```powershell
# WhisperX + spaCy 词级时间戳
conda run -p .conda\asr-env python src/processing/main.py ^
  input_audio\100k.mp4 ^
  --model medium ^
  --output_format source ^
  --segmentation_strategy spacy ^
  --word_timestamps

# 生成中英双语字幕
D:\Dream\translate\.conda\asr-env\python.exe -u src/processing/main.py ^
  input_audio\example.mp4 ^
  --model large-v3 ^
  --target_language zh-CN ^
  --output_format bilingual ^
  --word_timestamps
```
启用 `--word_timestamps` 可确保后续流程获取词级对齐信息，从而由 spaCy 对句子进行时间汇总。

## 处理流程概览
1. **音频提取** - 前缀内的 FFmpeg 将视频音频转为 WAV 分块。
2. **语音识别** - WhisperX 在 CUDA 12.1 上加载指定模型（默认 `medium`）。
3. **词级对齐** - WhisperX 输出每个词的起止时间。
4. **句法分割** - `src/processing/segmentation.py` 使用 spaCy 分句，必要时回退正则逻辑。
5. **句子对齐** - `src/processing/pipeline.py::generate_segments_with_alignment` 将句子映射到词序列，并计算时间。
6. **字幕封装** - `src/processing/main.py` 依据 `--max_subtitle_chars`、输出格式等参数生成最终字幕文件。

## 环境共享与更新
- 升级依赖后可重新导出：
  ```powershell
  conda env export -p D:\Dream\translate\.conda\asr-env --no-builds > environment\conda-env.yml
  conda run -p D:\Dream\translate\.conda\asr-env python -m pip freeze > requirements.txt
  ```
- 在其他机器上使用时，建议通过上述 YAML 创建新的 Conda 前缀；直接复制 `.conda\asr-env` 仅适用于相同系统与驱动。

## 常见问题
- **spaCy 模型缺失**：执行 `conda run -p .conda\asr-env python -m spacy download en_core_web_sm`。
- **GPU 不可用**：确认 NVIDIA 驱动满足 CUDA 12.1；若需 CPU 模式，请安装 CPU 版 PyTorch。
- **FFmpeg 相关报错**：务必通过 `start_gui.bat` 启动，以确保 PATH 指向前缀中的 DLL。
- **Pyannote 警告**：提示模型训练时的 Torch/pyannote 版本不同，通常可忽略，除非对齐效果明显下降。

## 项目结构
```
assets/               # 国际化资源
core/                 # GUI 数据流与处理调度
environment/          # 环境锁定文件
gui/                  # CustomTkinter 界面模块
input_audio/          # 默认输入目录
output_subtitles/     # 生成的字幕（按项目划分）
scripts/              # 环境/CLI 辅助脚本
src/                  # 核心处理流水线
start_gui.bat         # Windows GUI 启动器
```

## 许可证
详见 [LICENSE](LICENSE)。