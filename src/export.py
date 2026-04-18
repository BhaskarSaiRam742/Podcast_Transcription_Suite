"""
src/export.py
All export/serialisation helpers.
Keeps app.py clean — import these functions directly.
"""

import json
import csv
import io


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _srt_timestamp(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_json(result: dict, indent: int = 2) -> str:
    """Full result as pretty-printed JSON."""
    return json.dumps(result, indent=indent, ensure_ascii=False)


def to_srt(segments: list[dict]) -> str:
    """
    Standard SRT subtitle file.
    Each segment becomes one subtitle block.
    """
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = _srt_timestamp(seg["start"])
        end   = _srt_timestamp(seg["end"])
        # Wrap long lines at ~80 chars
        text = seg["text"]
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks)


def to_txt(result: dict) -> str:
    """Human-readable plain-text export with all segment metadata."""
    lines = [
        "PODCAST TRANSCRIPT",
        f"Source  : {result.get('source_label', 'N/A')}",
        f"Language: {result.get('language', 'unknown')}",
        f"Model   : {result.get('transcript_source', 'whisper')}",
        "=" * 70,
        "",
        "SUMMARY",
        result.get("summary", ""),
        "",
        "=" * 70,
        "",
    ]
    for seg in result["segments"]:
        lines += [
            f"[{fmt_time(seg['start'])} → {fmt_time(seg['end'])}]  "
            f"Segment #{seg['id']}  ·  {seg['topic']}",
            f"Duration : {seg['duration']:.0f}s",
            f"Keywords : {', '.join(seg['keywords'])}",
            f"Sentiment: {seg['sentiment']['label']} "
            f"(score {seg['sentiment']['score']:+.3f})",
            "",
            seg["text"],
            "",
            "-" * 70,
            "",
        ]
    return "\n".join(lines)


def to_csv(segments: list[dict]) -> str:
    """
    CSV with one row per segment — useful for spreadsheet analysis.
    Columns: id, topic, start, end, duration, sentiment_label,
             sentiment_score, keywords, summary, text
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)

    writer.writerow([
        "id", "topic", "start_s", "end_s", "duration_s",
        "sentiment_label", "sentiment_score",
        "keywords", "summary", "text"
    ])

    for seg in segments:
        writer.writerow([
            seg["id"],
            seg["topic"],
            round(seg["start"], 2),
            round(seg["end"], 2),
            round(seg["duration"], 2),
            seg["sentiment"]["label"],
            seg["sentiment"]["score"],
            "; ".join(seg["keywords"]),
            seg.get("summary", ""),
            seg["text"],
        ])

    return output.getvalue()


def to_markdown(result: dict) -> str:
    """
    Markdown report — good for sharing as a GitHub Gist or Notion page.
    """
    lines = [
        f"# 🎙️ Podcast Transcript",
        f"",
        f"**Source:** {result.get('source_label', 'N/A')}  ",
        f"**Language:** {result.get('language', 'unknown')}  ",
        f"**Segments:** {result.get('segment_count', 0)}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        result.get("summary", ""),
        f"",
        f"---",
        f"",
        f"## Segments",
        f"",
    ]

    for seg in result["segments"]:
        snt = seg["sentiment"]
        emoji = snt.get("emoji", "")
        kw = ", ".join(f"`{k}`" for k in seg["keywords"])
        lines += [
            f"### {seg['id']}. {seg['topic']}  {emoji} {snt['label']}",
            f"",
            f"⏱ `{fmt_time(seg['start'])}` → `{fmt_time(seg['end'])}`"
            f"  ·  {seg['duration']:.0f}s",
            f"",
            f"**Keywords:** {kw}",
            f"",
            seg["text"],
            f"",
            f"---",
            f"",
        ]

    return "\n".join(lines)
