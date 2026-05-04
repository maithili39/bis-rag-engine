"""Streamlit frontend for BIS Standards Recommendation Engine."""

import streamlit as st
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import DATASET_PDF

st.set_page_config(
    page_title="BIS Finder",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: #070707; }

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 4rem 2rem 4rem 2rem;
    max-width: 780px;
    margin: 0 auto;
}

/* ── Typography ── */
.wordmark {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #e8a020;
    display: block;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f0f0f0;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.6rem 0;
}
.hero-sub {
    font-size: 0.93rem;
    color: #5a5a5a;
    line-height: 1.5;
    margin-bottom: 2.5rem;
}

/* ── Search textarea ── */
.stTextArea > div > div > textarea {
    background: #101010 !important;
    border: 1.5px solid #222 !important;
    border-radius: 12px !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.94rem !important;
    line-height: 1.6 !important;
    padding: 0.9rem 1rem !important;
    caret-color: #e8a020;
    transition: border-color 0.2s;
    resize: none !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #e8a020 !important;
    box-shadow: 0 0 0 3px rgba(232,160,32,0.07) !important;
    outline: none !important;
}
.stTextArea label { display: none !important; }

/* ── Button ── */
.stButton > button {
    background: #e8a020 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    height: 46px !important;
    width: 100% !important;
    transition: background 0.15s, transform 0.1s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #f5b030 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #151515 !important; margin: 2rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { color: #e8a020 !important; }

/* ── Alert ── */
.stAlert { border-radius: 10px !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipeline = BISRAGPipeline()
    if not pipeline.load_existing_vectorstore():
        pipeline.initialize_from_pdf(DATASET_PDF)
    return pipeline


# ── Init pipeline ────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    with st.spinner("Loading knowledge base…"):
        try:
            st.session_state.pipeline = load_pipeline()
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.info("Run `python build_vectorstore.py` first, then restart.")
            st.stop()

if st.session_state.pipeline is None:
    st.error("Pipeline could not be initialised.")
    st.stop()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<span class="wordmark">Bureau of Indian Standards</span>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Standards Finder</h1>', unsafe_allow_html=True)
st.markdown(
    """<p class="hero-sub">Describe what your product does — we'll match it to the exact BIS standards you need to comply with.</p>""",
    unsafe_allow_html=True,
)

# ── Search ───────────────────────────────────────────────────────────────────
query = st.text_area(
    label="Product Description",
    label_visibility="collapsed",
    placeholder="e.g. We produce 33-grade Ordinary Portland Cement for structural use in residential buildings…",
    height=118,
    key="query_input",
)

col_btn, _, _ = st.columns([1, 1, 2])
with col_btn:
    search = st.button("Find Standards →", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
if search:
    if not query.strip():
        st.warning("Enter a product description first.")
    else:
        with st.spinner("Retrieving…"):
            codes, rationale, latency = st.session_state.pipeline.process_query(
                query.strip(), top_k=5
            )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Meta line
        st.markdown(
            f"<p style='color:#3a3a3a;font-size:0.78rem;margin-bottom:1.4rem;"
            f"font-family:JetBrains Mono,monospace;'>"
            f"{len(codes)} standards found &nbsp;·&nbsp; {latency:.2f} s</p>",
            unsafe_allow_html=True,
        )

        # Result cards
        for i, code in enumerate(codes, 1):
            rank_color = "#e8a020" if i == 1 else "#2e2e2e"
            text_color = "#f0f0f0" if i == 1 else "#b0b0b0"
            st.markdown(
                f"""
                <div style="
                    display:flex;align-items:center;gap:1rem;
                    padding:0.9rem 1.1rem;
                    border:1px solid {'#2a2a2a' if i == 1 else '#181818'};
                    border-radius:10px;margin-bottom:0.55rem;
                    background:{'#111' if i == 1 else '#0b0b0b'};
                ">
                    <span style="
                        font-family:'JetBrains Mono',monospace;
                        font-size:0.7rem;font-weight:600;
                        color:{rank_color};min-width:1.6rem;
                    ">#{i:02d}</span>
                    <span style="
                        font-family:'JetBrains Mono',monospace;
                        font-size:0.95rem;font-weight:500;
                        color:{text_color};flex:1;
                    ">{code}</span>
                    {'<span style="font-size:0.68rem;font-weight:700;background:rgba(232,160,32,0.12);color:#e8a020;padding:0.2rem 0.55rem;border-radius:4px;letter-spacing:0.06em;text-transform:uppercase;">Top match</span>' if i == 1 else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Rationale
        st.markdown(
            f"""
            <div style="
                margin-top:1.4rem;
                background:#0c0c0c;
                border:1px solid #1a1a1a;
                border-left:3px solid #e8a020;
                border-radius:8px;
                padding:0.85rem 1.1rem;
            ">
                <div style="
                    font-size:0.66rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.12em;
                    color:#e8a020;margin-bottom:0.4rem;
                ">Why these?</div>
                <div style="color:#666;font-size:0.87rem;line-height:1.65;">{rationale}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stats row
        st.markdown(
            f"""
            <div style="display:flex;gap:0.8rem;margin-top:1.4rem;">
                <div style="flex:1;background:#0c0c0c;border:1px solid #181818;border-radius:9px;
                            padding:0.8rem 1rem;text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                                font-weight:700;color:#f0f0f0;">{len(codes)}</div>
                    <div style="font-size:0.68rem;color:#444;text-transform:uppercase;
                                letter-spacing:0.08em;margin-top:0.2rem;">Standards</div>
                </div>
                <div style="flex:1;background:#0c0c0c;border:1px solid #181818;border-radius:9px;
                            padding:0.8rem 1rem;text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                                font-weight:700;color:#f0f0f0;">{latency:.2f}s</div>
                    <div style="font-size:0.68rem;color:#444;text-transform:uppercase;
                                letter-spacing:0.08em;margin-top:0.2rem;">Latency</div>
                </div>
                <div style="flex:1;background:#0c0c0c;border:1px solid #181818;border-radius:9px;
                            padding:0.8rem 1rem;text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                                font-weight:700;color:#22c55e;">✓</div>
                    <div style="font-size:0.68rem;color:#444;text-transform:uppercase;
                                letter-spacing:0.08em;margin-top:0.2rem;">Verified</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#252525;font-size:0.72rem;text-align:center;font-family:JetBrains Mono,monospace;'>"
    "BIS SP 21 Dataset &nbsp;·&nbsp; RAG + Hybrid Search &nbsp;·&nbsp; BIS × Startup Studio Hackathon 2026"
    "</p>",
    unsafe_allow_html=True,
)
