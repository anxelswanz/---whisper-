#!/usr/bin/env python3
"""
Re-translate the Chinese lines of an existing bilingual SRT so each Chinese
line stays aligned with its English fragment (fixes the cumulative drift caused
by translating mid-sentence fragments into whole sentences).

The English text and all timestamps are kept exactly as-is; only the second
(Chinese) line of each cue is regenerated.

Usage:
    python fix_zh_srt.py video.zh.srt
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("请先安装依赖: pip install openai")

TRANSLATION_BATCH_SIZE = 30
MODEL = "gpt-4o-mini"

CUE_RE = re.compile(
    r"(?P<index>\d+)\s*\n"
    r"(?P<time>\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(?P<body>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def parse_cues(text: str):
    cues = []
    for m in CUE_RE.finditer(text):
        body_lines = [l for l in m.group("body").splitlines() if l.strip() != ""]
        english = body_lines[0].strip() if body_lines else ""
        cues.append({"index": m.group("index"), "time": m.group("time"), "english": english})
    return cues


def translate_aligned(client: OpenAI, texts: list[str]) -> list[str]:
    """Translate each fragment in place, preserving 1:1 line alignment."""
    out: list[str] = []
    total = len(texts)
    for start in range(0, total, TRANSLATION_BATCH_SIZE):
        batch = texts[start : start + TRANSLATION_BATCH_SIZE]
        end = min(start + TRANSLATION_BATCH_SIZE, total)
        print(f"  翻译字幕 {start + 1}–{end}/{total}...")

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
        prompt = (
            "You are a professional subtitle translator. The following lines are "
            "consecutive subtitle fragments from one continuous narration. Many "
            "fragments begin or end mid-sentence.\n\n"
            "Translate each numbered line into 简体中文. CRITICAL ALIGNMENT RULES:\n"
            "1. Output exactly one Chinese line for every input line, with the same number.\n"
            "2. Line i's Chinese must correspond ONLY to the English content of line i. "
            "Do NOT pull words from the next line forward, and do NOT push leftover "
            "words to the next line. If a sentence is cut off, translate the partial "
            "fragment as a partial fragment (it is fine for a line to read as an "
            "incomplete clause).\n"
            "3. Use the surrounding lines only to understand meaning, never to "
            "redistribute content across lines.\n\n"
            'Return ONLY a JSON object with numeric string keys, e.g. {"1": "...", "2": "..."}.\n\n'
            f"{numbered}"
        )

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        batch_out = [str(parsed.get(str(i + 1), batch[i])) for i in range(len(batch))]
        out.extend(batch_out)
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit("用法: python fix_zh_srt.py <file.zh.srt>")
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    cues = parse_cues(text)
    if not cues:
        sys.exit("未能解析出字幕条目")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("请设置 OPENAI_API_KEY 环境变量")
    client = OpenAI(api_key=api_key)

    print(f"共 {len(cues)} 条字幕，重新对齐翻译中文...")
    zh = translate_aligned(client, [c["english"] for c in cues])

    blocks = []
    for i, c in enumerate(cues):
        blocks.append(f"{c['index']}\n{c['time']}\n{c['english']}\n{zh[i].strip()}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"已重写: {path}")


if __name__ == "__main__":
    main()
