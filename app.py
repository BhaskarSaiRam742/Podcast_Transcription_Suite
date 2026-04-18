"""
app.py  — Podcast Transcription Suite
Run with:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Podcast Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

    .stApp { background: #0d0f14; color: #e8e8f0; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .segment-card {
        background: #161a24;
        border: 1px solid #2a2e3e;
        border-left: 3px solid #a78bfa;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .segment-topic {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: #a78bfa;
        margin-bottom: 0.3rem;
    }

    .segment-meta {
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }

    .segment-text {
        font-size: 0.9rem;
        line-height: 1.7;
        color: #d1d5db;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .sentiment-positive { background: #064e3b; color: #34d399; }
    .sentiment-negative { background: #4c0519; color: #f87171; }
    .sentiment-neutral  { background: #1e3a5f; color: #93c5fd; }

    .keyword-pill {
        display: inline-block;
        background: #1e2030;
        border: 1px solid #374151;
        color: #9ca3af;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.73rem;
        margin: 2px;
    }

    .summary-box {
        background: linear-gradient(135deg, #13161f, #1a1e2e);
        border: 1px solid #2a2e3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .stat-card {
        background: #161a24;
        border: 1px solid #2a2e3e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #a78bfa;
    }
    .stat-label { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }

    .source-badge {
        background: #1a2740;
        border: 1px solid #1e4080;
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
    }

    div[data-testid="stProgress"] > div > div { background: #a78bfa !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    div[data-testid="stFileUploader"] {
        background: #161a24 !important;
        border: 1.5px dashed #374151 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def sentiment_badge_html(sentiment: dict) -> str:
    label = sentiment.get("label", "Neutral")
    emoji = sentiment.get("emoji", "😐")
    cls = f"sentiment-{label.lower()}"
    return f'<span class="sentiment-badge {cls}">{emoji} {label}</span>'


def keywords_html(keywords: list) -> str:
    pills = "".join(f'<span class="keyword-pill">{k}</span>' for k in keywords)
    return f'<div style="margin-top:0.5rem">{pills}</div>'


from src.export import to_json, to_srt, to_txt, to_csv, to_markdown


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # Model selector with speed/accuracy guide
    st.markdown("**🧠 Whisper Model**")
    model_choice = st.radio(
        "model",
        options=["tiny", "base", "small"],
        index=0,
        label_visibility="collapsed",
        help="tiny=fastest, small=most accurate"
    )
    model_info = {
        "tiny":  ("⚡ ~2–4 min / 30-min podcast", "#34d399"),
        "base":  ("⚖️ ~4–7 min / 30-min podcast", "#60a5fa"),
        "small": ("🎯 ~8–12 min / 30-min podcast", "#a78bfa"),
    }
    info_text, info_color = model_info[model_choice]
    st.markdown(
        f'<div style="background:#161a24;border-left:3px solid {info_color};'
        f'padding:6px 10px;border-radius:4px;font-size:0.8rem;color:{info_color};'
        f'margin-bottom:0.8rem">{info_text}</div>',
        unsafe_allow_html=True
    )

    # Write model choice to transcript.py dynamically
    import importlib, sys
    if "src.transcript" in sys.modules:
        import src.transcript as _t
        if _t._MODEL_SIZE != model_choice:
            _t._MODEL_SIZE = model_choice
            _t._model = None   # force reload on next run

    st.markdown("---")

    use_semantic = st.toggle(
        "Semantic segmentation",
        value=False,
        help="Uses sentence-transformers to detect topic shifts. Slower but smarter."
    )
    use_transformer_sentiment = st.toggle(
        "Advanced sentiment (distilbert)",
        value=False,
        help="More accurate, but requires ~250 MB download on first run."
    )
    whisper_lang = st.selectbox(
        "Language (leave blank = auto-detect)",
        ["Auto", "en", "es", "fr", "de", "hi", "zh", "ja", "ar", "pt"],
        index=0
    )
    whisper_lang = None if whisper_lang == "Auto" else whisper_lang

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Supports uploaded audio/video files and URLs "
        "(YouTube, direct mp3/mp4, podcast RSS feeds)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎙️ Podcast Transcription Suite</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#6b7280;margin-top:0.3rem;'>Upload a file or paste a URL — "
    "YouTube, direct audio/video links, or podcast RSS feeds.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Input area — tabs for File vs URL
# ─────────────────────────────────────────────────────────────────────────────
tab_file, tab_url = st.tabs(["📁  Upload File", "🔗  Paste URL"])

source = None
source_type = None

with tab_file:
    uploaded = st.file_uploader(
        "Drop your audio or video file here",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "mp4", "mkv", "avi", "mov", "webm"],
        label_visibility="collapsed"
    )
    if uploaded:
        source = uploaded
        source_type = "file"

with tab_url:
    url_input = st.text_input(
        "YouTube URL, direct media URL, or podcast RSS feed",
        placeholder="https://www.youtube.com/watch?v=... or https://example.com/podcast.mp3",
        label_visibility="collapsed"
    )
    if url_input.strip():
        source = url_input.strip()
        source_type = "url"

# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("")
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶  Run Pipeline", disabled=(source is None))

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and source is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(msg, pct):
        progress_bar.progress(pct)
        status_text.markdown(f"**{msg}**")

    try:
        from src.pipeline import run_pipeline

        os.makedirs("audio_processed", exist_ok=True)

        if source_type == "file":
            # Save uploaded file to disk
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="audio_processed") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            result = run_pipeline(
                source=tmp_path,
                source_type="file",
                use_semantic_segmentation=use_semantic,
                use_transformer_sentiment=use_transformer_sentiment,
                whisper_language=whisper_lang,
                output_dir="audio_processed",
                progress_cb=progress_cb,
            )
        else:
            result = run_pipeline(
                source=source,
                source_type="url",
                use_semantic_segmentation=use_semantic,
                use_transformer_sentiment=use_transformer_sentiment,
                whisper_language=whisper_lang,
                output_dir="audio_processed",
                progress_cb=progress_cb,
            )

        progress_bar.empty()
        status_text.empty()
        st.session_state["result"] = result

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Pipeline failed: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    segs = result["segments"]

    # Source badge
    st.markdown(f'<div class="source-badge">📡 {result["source_label"]}</div>', unsafe_allow_html=True)

    # ── Stats row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_duration = segs[-1]["end"] if segs else 0
    pos = sum(1 for s in segs if s["sentiment"]["label"] == "Positive")
    neg = sum(1 for s in segs if s["sentiment"]["label"] == "Negative")

    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{result["segment_count"]}</div>'
                    f'<div class="stat-label">Segments</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{fmt_time(total_duration)}</div>'
                    f'<div class="stat-label">Total Duration</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#34d399">{pos}</div>'
                    f'<div class="stat-label">Positive Segments</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#f87171">{neg}</div>'
                    f'<div class="stat-label">Negative Segments</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Summary ──────────────────────────────────────────────────────────────
    with st.expander("📋 Overall Summary", expanded=True):
        st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
        if result.get("language") and result["language"] != "unknown":
            st.caption(f"Detected language: **{result['language']}** · "
                       f"Transcript source: **{result['transcript_source']}**")

    # ── Segments ─────────────────────────────────────────────────────────────
    st.markdown("### 🔀 Transcript Segments")

    filter_col, _ = st.columns([1, 3])
    with filter_col:
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Neutral", "Negative"],
            label_visibility="collapsed"
        )

    displayed_segs = segs if sentiment_filter == "All" else [
        s for s in segs if s["sentiment"]["label"] == sentiment_filter
    ]

    for seg in displayed_segs:
        snt = seg["sentiment"]
        badge = sentiment_badge_html(snt)
        kw_html = keywords_html(seg["keywords"])

        card = f"""
        <div class="segment-card">
            <div class="segment-topic">{seg['topic']}{badge}</div>
            <div class="segment-meta">
                ⏱ {fmt_time(seg['start'])} → {fmt_time(seg['end'])}
                &nbsp;·&nbsp; {seg['duration']:.0f}s
                &nbsp;·&nbsp; Segment #{seg['id']}
            </div>
            <div class="segment-text">{seg['text']}</div>
            {kw_html}
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

        with st.expander(f"📌 Summary — Segment #{seg['id']}"):
            st.write(seg["summary"])

    # ── Full transcript ───────────────────────────────────────────────────────
    with st.expander("📄 Full Raw Transcript"):
        st.text_area("", value=result["full_text"], height=300, label_visibility="collapsed")

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("### 💾 Export")
    dl1, dl2, dl3, dl4, dl5 = st.columns(5)

    with dl1:
        st.download_button("⬇ JSON", data=to_json(result),
                           file_name="transcript.json", mime="application/json")
    with dl2:
        st.download_button("⬇ SRT", data=to_srt(segs),
                           file_name="transcript.srt", mime="text/plain")
    with dl3:
        st.download_button("⬇ TXT", data=to_txt(result),
                           file_name="transcript.txt", mime="text/plain")
    with dl4:
        st.download_button("⬇ CSV", data=to_csv(segs),
                           file_name="transcript.csv", mime="text/csv")
    with dl5:
        st.download_button("⬇ Markdown", data=to_markdown(result),
                           file_name="transcript.md", mime="text/markdown")
