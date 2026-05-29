# 视频字幕生成 SOP

使用 OpenAI Whisper API 将视频转录为 SRT 字幕，支持单语和双语模式。

---

## 一、环境准备（首次执行）

### 1. 安装 Python 依赖

```bash
pip install openai
```

### 2. 安装 ffmpeg（用于从视频中提取音频）

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows（使用 winget）
winget install ffmpeg
```

验证安装：

```bash
ffmpeg -version
ffprobe -version
```

### 3. 配置 OpenAI API Key

```bash
# 临时设置（当前终端会话有效）
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 永久设置（写入 shell 配置文件）
echo 'export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

---

## 二、使用说明

### 基础用法：单语字幕

```bash
# 自动检测语言
python transcribe.py video.mp4

# 指定音频语言（推荐，速度更快、更准确）
python transcribe.py video.mp4 --language en   # 英文视频
python transcribe.py video.mp4 --language zh   # 中文视频
python transcribe.py video.mp4 --language ja   # 日文视频

# 处理整个目录下的所有视频
python transcribe.py ./videos/

# 指定 SRT 输出目录
python transcribe.py video.mp4 --output-dir ./subtitles/
```

### 双语字幕

```bash
# 英文视频 → 英文 + 中文（默认目标语言为中文）
python transcribe.py video.mp4 --language en --bilingual

source .env && python3 transcribe.py video.MOV --language en --bilingual

# 中文视频 → 中文 + 英文
python transcribe.py video.mp4 --language en --bilingual --target-language zh

# 日文视频 → 日文 + 中文
python transcribe.py video.mp4 --language ja --bilingual

# 批量处理目录 + 双语
python transcribe.py ./videos/ --language en --bilingual --output-dir ./subtitles/
```

### 所有参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `target` | — | 必填 | 视频文件路径或目录路径 |
| `--language` | `-l` | 自动检测 | 音频原始语言代码 |
| `--bilingual` | `-b` | 关闭 | 开启双语字幕模式 |
| `--target-language` | `-t` | `zh` | 翻译目标语言代码 |
| `--translation-model` | — | `gpt-4o-mini` | 翻译使用的 GPT 模型 |
| `--output-dir` | `-o` | 视频同目录 | SRT 文件输出目录 |
| `--api-key` | — | 读取环境变量 | OpenAI API Key |

### 语言代码参考

| 语言 | 代码 |
|------|------|
| 中文（简体） | `zh` |
| 中文（繁体） | `zh-tw` |
| 英文 | `en` |
| 日文 | `ja` |
| 韩文 | `ko` |
| 法文 | `fr` |
| 德文 | `de` |
| 西班牙文 | `es` |

---

## 三、输出文件说明

| 模式 | 输出文件名示例 |
|------|--------------|
| 单语 | `video.srt` |
| 双语（目标中文） | `video.zh.srt` |
| 双语（目标英文） | `video.en.srt` |

双语 SRT 格式示例：

```
1
00:00:01,200 --> 00:00:04,800
Welcome to this tutorial on machine learning.
欢迎来到这个机器学习教程。

2
00:00:05,100 --> 00:00:08,300
Today we'll cover neural networks.
今天我们将介绍神经网络。
```

---

## 四、费用参考

### 计费规则

- **Whisper API**：按音频时长计费，`$0.006 / 分钟`
- **gpt-4o-mini 翻译**：按 tokens 计费，`$0.15 / 1M input tokens`，`$0.60 / 1M output tokens`

### 不同时长的估算费用

| 视频时长 | Whisper | gpt-4o-mini 翻译 | 合计（约） |
|----------|---------|------------------|-----------|
| 10 分钟  | $0.06   | $0.001           | **$0.06** |
| 30 分钟  | $0.18   | $0.003           | **$0.18** |
| 60 分钟  | $0.36   | $0.006           | **$0.37** |
| 120 分钟 | $0.72   | $0.012           | **$0.73** |

> 翻译费用可忽略不计，主要成本来自 Whisper 转录。

---

## 五、支持的视频 / 音频格式

| 类型 | 格式 |
|------|------|
| 视频 | `.mp4` `.mkv` `.mov` `.avi` `.flv` `.webm` `.m4v` `.wmv` `.ts` `.m2ts` |
| 音频 | `.mp3` `.wav` `.m4a` `.flac` `.ogg` `.aac` |

---

## 六、常见问题

### Q: 提示 `OPENAI_API_KEY` 未设置

```bash
export OPENAI_API_KEY="sk-xxxx"
```

或在命令中直接传入：

```bash
python transcribe.py video.mp4 --api-key sk-xxxx
```

### Q: 提示 `ffmpeg: command not found`

按照「一、环境准备」中的步骤安装 ffmpeg。

### Q: 文件超过 25MB 限制

脚本会自动分段处理，每段 10 分钟，无需手动操作。

### Q: 转录准确率不高

指定 `--language` 参数可显著提升准确率：

```bash
python transcribe.py video.mp4 --language en
```

### Q: 翻译结果不自然

换用更强的翻译模型：

```bash
python transcribe.py video.mp4 --bilingual --translation-model gpt-4o
```

### Q: 如何在播放器中加载字幕

将 `.srt` 文件放在与视频**同一目录**，并确保文件名与视频相同（扩展名改为 `.srt`）。大多数播放器（VLC、IINA、PotPlayer）会自动识别加载。

---

## 七、完整示例流程

```bash
# 1. 进入脚本目录
cd /Users/ronghuizhong/Documents/project/videos/whisper

# 2. 设置 API Key
export OPENAI_API_KEY="sk-xxxx"

# 3. 单个英文视频 → 双语字幕
python transcribe.py ~/Movies/lecture.mp4 --language en --bilingual

# 4. 批量处理整个 videos 目录 → 双语字幕输出到 subtitles 目录
python transcribe.py ~/Movies/videos/ --language en --bilingual --output-dir ~/Movies/subtitles/
```
