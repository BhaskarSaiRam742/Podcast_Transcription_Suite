"""
url_handler.py
Handles all URL-based input:
  - YouTube: tries to fetch existing captions first, falls back to audio download
  - Direct audio/video URLs (.mp3, .mp4, .wav, etc.): downloads the file
  - Podcast RSS feeds: finds the latest episode and downloads it
"""

import os
import re
import requests
import feedparser


def is_youtube_url(url: str) -> bool:
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=",
        r"(?:https?://)?(?:www\.)?youtu\.be/",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/",
    ]
    return any(re.search(p, url) for p in patterns)


def is_direct_media_url(url: str) -> bool:
    media_extensions = (".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm")
    return any(url.lower().split("?")[0].endswith(ext) for ext in media_extensions)


def is_rss_feed(url: str) -> bool:
    """Try to detect if the URL is a podcast RSS feed."""
    try:
        feed = feedparser.parse(url)
        return len(feed.entries) > 0 and hasattr(feed.entries[0], 'enclosures')
    except Exception:
        return False


def fetch_youtube_transcript(url: str):
    """
    Attempt to fetch existing YouTube captions using multiple fallback strategies.
    Tries: manual captions → auto-generated → any available language.
    Returns a transcript dict if found, else None.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
        video_id_match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not video_id_match:
            return None

        video_id = video_id_match.group(1)

        # Strategy 1: try English manual captions
        # Strategy 2: try auto-generated captions
        # Strategy 3: try any available language
        transcript_list = None
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except Exception:
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                # Try auto-generated first, then manual
                try:
                    t = transcripts.find_generated_transcript(["en"])
                    transcript_list = t.fetch()
                except Exception:
                    # Just grab whatever is available
                    t = next(iter(transcripts))
                    transcript_list = t.fetch()
            except Exception:
                return None

        if not transcript_list:
            return None

        segments = [
            {
                "start": entry["start"],
                "end":   entry["start"] + entry["duration"],
                "text":  entry["text"].strip()
            }
            for entry in transcript_list
        ]
        full_text = " ".join(s["text"] for s in segments)

        return {
            "full_text": full_text,
            "summary":   "",
            "segments":  segments,
            "source":    "youtube_captions"
        }
    except Exception:
        return None


def download_youtube_audio(url: str, output_dir: str = "audio_processed") -> str:
    """
    Download YouTube audio using yt-dlp with cloud-friendly options.
    Uses browser impersonation to bypass YouTube's bot detection (403 errors).
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    # Base options with browser impersonation
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "128",     # lower quality = faster download
        }],
        "quiet": True,
        "no_warnings": True,
        # Bypass bot detection
        "extractor_args": {"youtube": {"player_client": ["web", "android"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        "socket_timeout": 60,
        "retries": 5,
        "fragment_retries": 5,
    }

    last_error = None

    # Try multiple player clients in order until one works
    for client in [["web"], ["android"], ["ios"], ["web_embedded"]]:
        try:
            opts = dict(ydl_opts)
            opts["extractor_args"] = {"youtube": {"player_client": client}}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get("id", "audio")
            return os.path.join(output_dir, f"{video_id}.wav")
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not download YouTube audio after trying all methods.\n"
        f"Last error: {last_error}\n\n"
        f"💡 Tip: YouTube videos with captions work best on cloud deployment. "
        f"Try a video that has auto-generated subtitles, or upload the audio file directly."
    )


def download_direct_url(url: str, output_dir: str = "audio_processed") -> str:
    """Download a direct audio/video URL and return local path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split("?")[0].split("/")[-1]
    output_path = os.path.join(output_dir, filename)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path


def download_rss_episode(url: str, output_dir: str = "audio_processed", episode_index: int = 0) -> str:
    """Parse a podcast RSS feed and download the specified episode (default: latest)."""
    feed = feedparser.parse(url)
    if not feed.entries:
        raise ValueError("No episodes found in RSS feed.")

    entry = feed.entries[episode_index]
    enclosures = entry.get("enclosures", [])
    if not enclosures:
        raise ValueError("No audio enclosure found in this RSS episode.")

    audio_url = enclosures[0].get("href") or enclosures[0].get("url")
    if not audio_url:
        raise ValueError("Could not extract audio URL from RSS feed.")

    return download_direct_url(audio_url, output_dir)


def resolve_url(url: str, output_dir: str = "audio_processed") -> dict:
    """
    Master resolver. Given any URL, returns:
      {
        "type":         "transcript" | "audio_file",
        "data":         <transcript dict>  -- if type == "transcript",
        "file_path":    <str>              -- if type == "audio_file",
        "source_label": <str>             -- human-readable description
      }
    """
    url = url.strip()

    if is_youtube_url(url):
        transcript = fetch_youtube_transcript(url)
        if transcript:
            return {"type": "transcript", "data": transcript,
                    "source_label": "YouTube (existing captions found ✅)"}
        file_path = download_youtube_audio(url, output_dir)
        return {"type": "audio_file", "file_path": file_path,
                "source_label": "YouTube (no captions — audio downloaded)"}

    if is_direct_media_url(url):
        file_path = download_direct_url(url, output_dir)
        return {"type": "audio_file", "file_path": file_path,
                "source_label": "Direct media URL"}

    if is_rss_feed(url):
        file_path = download_rss_episode(url, output_dir)
        return {"type": "audio_file", "file_path": file_path,
                "source_label": "Podcast RSS feed (latest episode)"}

    raise ValueError(
        "Unrecognised URL. Please provide a YouTube link, "
        "direct audio/video URL (.mp3/.mp4/etc.), or a podcast RSS feed URL."
    )
