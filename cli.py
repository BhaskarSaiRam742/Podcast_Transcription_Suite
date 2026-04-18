#!/usr/bin/env python3
"""
cli.py  —  Command-line interface for the podcast transcription pipeline.

Usage examples:
  # Transcribe a local file
  python cli.py --file my_podcast.mp3

  # Transcribe from a YouTube URL
  python cli.py --url https://www.youtube.com/watch?v=xxxxx

  # Full options
  python cli.py --file episode.mp4 --output results/ --format json srt txt md

  # Use semantic segmentation and advanced sentiment
  python cli.py --url https://... --semantic --transformer-sentiment
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎙️ Podcast Transcription Suite — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", metavar="PATH",
                        help="Path to a local audio or video file")
    source.add_argument("--url", metavar="URL",
                        help="YouTube URL, direct media URL, or podcast RSS feed")

    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--format", nargs="+",
                        choices=["json", "srt", "txt", "csv", "md"],
                        default=["json", "txt"],
                        help="Export formats (default: json txt)")
    parser.add_argument("--language", default=None,
                        help="ISO 639-1 language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--semantic", action="store_true",
                        help="Enable semantic segmentation (needs sentence-transformers)")
    parser.add_argument("--transformer-sentiment", action="store_true",
                        help="Use distilbert for sentiment (needs transformers)")
    parser.add_argument("--audio-dir", default="audio_processed",
                        help="Temp dir for processed audio (default: audio_processed)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")

    return parser.parse_args()


def main():
    args = parse_args()

    # Progress callback
    def progress(msg, pct):
        if not args.quiet:
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r[{bar}] {pct:3d}%  {msg}", end="", flush=True)
            if pct == 100:
                print()

    # Determine source
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        source = args.file
        source_type = "file"
    else:
        source = args.url
        source_type = "url"

    # Run pipeline
    from src.pipeline import run_pipeline

    if not args.quiet:
        print(f"\n🎙️  Podcast Transcription Suite")
        print(f"   Source : {source}")
        print(f"   Formats: {', '.join(args.format)}\n")

    try:
        result = run_pipeline(
            source=source,
            source_type=source_type,
            use_semantic_segmentation=args.semantic,
            use_transformer_sentiment=args.transformer_sentiment,
            whisper_language=args.language,
            output_dir=args.audio_dir,
            progress_cb=progress,
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    os.makedirs(args.output, exist_ok=True)

    from src.export import to_json, to_srt, to_txt, to_csv, to_markdown

    written = []
    for fmt in args.format:
        if fmt == "json":
            path = os.path.join(args.output, "transcript.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_json(result))
        elif fmt == "srt":
            path = os.path.join(args.output, "transcript.srt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_srt(result["segments"]))
        elif fmt == "txt":
            path = os.path.join(args.output, "transcript.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_txt(result))
        elif fmt == "csv":
            path = os.path.join(args.output, "transcript.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(to_csv(result["segments"]))
        elif fmt == "md":
            path = os.path.join(args.output, "transcript.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_markdown(result))
        written.append(path)

    if not args.quiet:
        print(f"\n✅ Done! {len(result['segments'])} segments · {len(result['full_text'].split())} words")
        print(f"\n📁 Output files:")
        for p in written:
            print(f"   {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
