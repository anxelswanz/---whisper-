#!/usr/bin/env python3
"""
Video to SRT subtitle generator using OpenAI Whisper API (whisper-large-v3).
Supports bilingual subtitles via GPT translation.

Usage:
    python transcribe.py <video_file_or_directory> [options]

Examples:
    python transcribe.py video.mp4
    python transcribe.py ./videos/
    python transcribe.py video.mp4 --language en
    python transcribe.py video.mp4 --bilingual          # 原文 + 中文
    python transcribe.py video.mp4 --bilingual --target-language en  # 原文 + 英文
    python transcribe.py video.mp4 --output-dir ./subtitles
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("请先安装依赖: pip install openai")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v", ".wmv", ".ts", ".m2ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
MAX_FILE_SIZE_MB = 25
TRANSLATION_BATCH_SIZE = 30  # 每次翻译的字幕条数


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict], translations: list[str] | None = None) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        original = seg["text"].strip()
        if translations and i - 1 < len(translations):
            translated = translations[i - 1].strip()
            text = f"{original}\n{translated}"
        else:
            text = original
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def translate_segments(client: OpenAI, segments: list[dict], target_language: str, model: str) -> list[str]:
    """Batch translate all segment texts using GPT."""
    texts = [seg["text"].strip() for seg in segments]
    translations: list[str] = []

    lang_name = {
        "zh": "简体中文",
        "zh-tw": "繁体中文",
        "en": "English",
        "ja": "日本語",
        "ko": "한국어",
        "fr": "Français",
        "de": "Deutsch",
        "es": "Español",
    }.get(target_language, target_language)

    total = len(texts)
    for batch_start in range(0, total, TRANSLATION_BATCH_SIZE):
        batch = texts[batch_start : batch_start + TRANSLATION_BATCH_SIZE]
        batch_end = min(batch_start + TRANSLATION_BATCH_SIZE, total)
        print(f"  翻译字幕 {batch_start + 1}–{batch_end}/{total}...")

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
        prompt = (
            f"You are a professional subtitle translator. "
            f"Translate the following {len(batch)} subtitle lines into {lang_name}. "
            f'Return ONLY a JSON object with numeric string keys, e.g. {{"1": "...", "2": "...", ...}}. '
            f"Include every line. Keep translations concise and natural for subtitles.\n\n"
            f"{numbered}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                batch_translations = [str(t) for t in parsed]
            elif all(str(k).isdigit() for k in parsed):
                # {"1": "...", "2": "..."} keyed by index
                batch_translations = [str(parsed[str(i + 1)]) for i in range(len(batch)) if str(i + 1) in parsed]
            else:
                # {"translations": [...]} or similar wrapper
                inner = next(iter(parsed.values()))
                batch_translations = [str(t) for t in inner] if isinstance(inner, list) else batch[:]
            # Pad if GPT returned fewer lines than expected
            if len(batch_translations) < len(batch):
                print(f"  警告: 返回条数 {len(batch_translations)} 少于输入 {len(batch)}，缺失行保留原文")
                batch_translations += batch[len(batch_translations):]
            batch_translations = batch_translations[: len(batch)]
        except Exception as e:
            print(f"  警告: 翻译解析失败 ({e})，该批次使用原文")
            batch_translations = batch[:]

        translations.extend(str(t) for t in batch_translations)

    return translations


def extract_audio(video_path: Path, output_path: Path) -> bool:
    ret = os.system(
        f'ffmpeg -y -i "{video_path}" -vn -ar 16000 -ac 1 -c:a pcm_s16le "{output_path}" -loglevel error'
    )
    return ret == 0


def transcribe_file(client: OpenAI, audio_path: Path, language: str | None) -> list[dict]:
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"  警告: 文件 {file_size_mb:.1f}MB 超过 {MAX_FILE_SIZE_MB}MB 限制，分段处理...")
        return transcribe_chunked(client, audio_path, language)

    with open(audio_path, "rb") as f:
        kwargs = dict(model="whisper-1", file=f, response_format="verbose_json", timestamp_granularities=["segment"])
        if language:
            kwargs["language"] = language
        response = client.audio.transcriptions.create(**kwargs)

    return [{"start": s.start, "end": s.end, "text": s.text} for s in response.segments]


def transcribe_chunked(client: OpenAI, audio_path: Path, language: str | None) -> list[dict]:
    import math

    probe_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{audio_path}"'
    duration_str = os.popen(probe_cmd).read().strip()
    try:
        total_duration = float(duration_str)
    except ValueError:
        sys.exit("错误: 无法获取音频时长，请确认 ffprobe 已安装")

    chunk_duration = 10 * 60
    all_segments: list[dict] = []
    num_chunks = math.ceil(total_duration / chunk_duration)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(num_chunks):
            start_sec = i * chunk_duration
            chunk_path = Path(tmp_dir) / f"chunk_{i:04d}.wav"
            os.system(
                f'ffmpeg -y -i "{audio_path}" -ss {start_sec} -t {chunk_duration} '
                f'-ar 16000 -ac 1 -c:a pcm_s16le "{chunk_path}" -loglevel error'
            )
            print(f"  处理分段 {i+1}/{num_chunks}...")
            segs = transcribe_file(client, chunk_path, language)
            for seg in segs:
                seg["start"] += start_sec
                seg["end"] += start_sec
            all_segments.extend(segs)

    return all_segments


def process_video(
    client: OpenAI,
    video_path: Path,
    output_dir: Path,
    language: str | None,
    bilingual: bool,
    target_language: str,
    translation_model: str,
) -> None:
    suffix = f".{target_language}.srt" if bilingual else ".srt"
    srt_path = output_dir / (video_path.stem + suffix)
    print(f"处理: {video_path.name}")

    is_audio = video_path.suffix.lower() in AUDIO_EXTENSIONS

    with tempfile.TemporaryDirectory() as tmp_dir:
        if is_audio:
            audio_path = video_path
        else:
            audio_path = Path(tmp_dir) / "audio.wav"
            print(f"  提取音频...")
            if not extract_audio(video_path, audio_path):
                print(f"  错误: 提取音频失败，跳过")
                return

        print(f"  调用 Whisper API (whisper-large-v3)...")
        try:
            segments = transcribe_file(client, audio_path, language)
        except Exception as e:
            print(f"  错误: 转录失败 — {e}")
            return

    if not segments:
        print(f"  警告: 未识别到任何内容")
        return

    translations = None
    if bilingual:
        print(f"  使用 {translation_model} 翻译为 {target_language}...")
        try:
            translations = translate_segments(client, segments, target_language, translation_model)
        except Exception as e:
            print(f"  错误: 翻译失败 — {e}，将只输出原文")

    srt_content = segments_to_srt(segments, translations)
    srt_path.write_text(srt_content, encoding="utf-8")
    print(f"  已生成: {srt_path}")


def collect_files(target: Path) -> list[Path]:
    all_exts = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
    if target.is_file():
        if target.suffix.lower() in all_exts:
            return [target]
        sys.exit(f"不支持的文件类型: {target.suffix}")
    elif target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix.lower() in all_exts)
        if not files:
            sys.exit(f"目录中未找到视频/音频文件: {target}")
        return files
    else:
        sys.exit(f"路径不存在: {target}")


def main():
    parser = argparse.ArgumentParser(description="使用 OpenAI Whisper 将视频转为 SRT 字幕，支持双语")
    parser.add_argument("target", help="视频文件或包含视频的目录")
    parser.add_argument("--language", "-l", default=None, help="音频原始语言代码，如 en、zh、ja（留空自动检测）")
    parser.add_argument("--bilingual", "-b", action="store_true", help="生成双语字幕（原文 + 译文）")
    parser.add_argument("--target-language", "-t", default="zh", help="翻译目标语言代码，默认 zh（简体中文）")
    parser.add_argument("--translation-model", default="gpt-4o-mini", help="翻译使用的 GPT 模型，默认 gpt-4o-mini")
    parser.add_argument("--output-dir", "-o", default=None, help="SRT 输出目录（默认与视频同目录）")
    parser.add_argument("--api-key", default=None, help="OpenAI API Key（也可通过 OPENAI_API_KEY 环境变量设置）")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("请设置 OPENAI_API_KEY 环境变量或通过 --api-key 传入")

    client = OpenAI(api_key=api_key)
    target = Path(args.target).expanduser().resolve()
    files = collect_files(target)

    print(f"找到 {len(files)} 个文件待处理")
    if args.bilingual:
        print(f"双语模式: 原文 + {args.target_language}（翻译模型: {args.translation_model}）")
    print()

    for video_path in files:
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else video_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        process_video(
            client,
            video_path,
            output_dir,
            args.language,
            args.bilingual,
            args.target_language,
            args.translation_model,
        )

    print("\n全部完成。")


if __name__ == "__main__":
    main()
