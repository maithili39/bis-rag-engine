import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add src to path so internal src imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_pipeline import BISRAGPipeline
from src.config import CHROMA_PERSIST_DIR, DATASET_PDF

# Page Configuration
st.set_page_config(
    page_title="BIS Standards Recommendation",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium Styling (Aurora / Violet-Cyan Theme)
st.markdown("""
<style>
    /* Hide Default Streamlit Style Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }

    :root {
        --bg-deep: #ffffff;
        --bg-panel: #f8f9fa;
        --accent: #b45309;
        --accent-2: #92400e;
        --accent-tint: #fef3e2;
        --text-dim: #6b7280;
        --border-color: rgba(0,0,0,0.08);
        --white: #ffffff;
    }

    [data-testid="stAppViewContainer"], .stApp {
        background: #ffffff !important;
        color: #1f2937;
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Top Gov Bar */
    .gov-bar {
        background: #f0f1f3;
        color: #6b7280;
        padding: 0.5rem 2rem;
        font-size: 0.72rem;
        font-weight: 500;
        border-bottom: 1px solid #e5e7eb;
        margin-top: -6rem;
        margin-bottom: 2rem;
        margin-left: -5rem;
        margin-right: -5rem;
        letter-spacing: 0.03em;
    }

    /* Header Banner */
    .top-header {
        position: relative;
        overflow: hidden;
        background: #f9fafb;
        color: #1f2937;
        padding: 2.2rem 2rem;
        border-bottom: 1px solid #e5e7eb;
        margin-left: -5rem;
        margin-right: -5rem;
        margin-top: -2.1rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .top-header::before {
        display: none;
    }

    .top-header::after {
        display: none;
    }

    .logo-area {
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        z-index: 1;
    }

    .logo-text h1 {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        color: #1f2937;
        letter-spacing: -0.01em;
    }

    .logo-text p {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        margin: 0.2rem 0 0 0;
        letter-spacing: 0.08em;
    }

    /* Navigation Bar */
    .main-nav {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        margin-left: -5rem;
        margin-right: -5rem;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
    }

    .main-nav ul {
        list-style: none;
        display: flex;
        padding: 0 2rem;
        margin: 0;
    }

    .main-nav li {
        padding: 0.9rem 1.5rem;
        color: #6b7280;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        position: relative;
    }

    .main-nav li.active {
        color: #1f2937;
        border-bottom: 2px solid var(--accent);
    }

    /* Breadcrumbs */
    .breadcrumbs {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    .breadcrumbs .current {
        color: #6b7280;
        font-weight: 600;
    }

    /* Title & Badge */
    .title-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .title-section h2 {
        font-size: 2rem;
        font-weight: 800;
        color: #1f2937;
        margin: 0;
        letter-spacing: -0.01em;
    }

    .badge {
        background: var(--accent-tint);
        color: var(--accent-2);
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.3rem 0.7rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
        border: 1px solid #f3d9ae;
    }

    /* Scope Info Banner */
    .scope-banner {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-left: 4px solid var(--accent);
        border-radius: 4px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.5rem;
        font-size: 0.84rem;
        color: #374151;
        line-height: 1.6;
    }
    .scope-banner strong {
        color: #1f2937;
    }
    .scope-banner .scope-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.6rem;
    }
    .scope-tag {
        background: #e5e7eb;
        color: #374151;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        letter-spacing: 0.03em;
        border: 1px solid #d1d5db;
    }

    /* Sample queries row */
    .sample-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Streamlit native buttons -> pill style matching theme */
    .stButton > button {
        background: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        transition: border-color 0.15s ease, color 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent-2) !important;
        box-shadow: none !important;
        transform: none;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-2) !important;
        box-shadow: none !important;
        transform: none;
    }

    /* Text area */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-tint) !important;
    }
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }

    /* Card Styling */
    .standard-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        transition: border-color 0.15s ease;
    }

    .standard-card:hover {
        border-color: var(--accent);
    }

    .rank-badge {
        font-size: 0.85rem;
        font-weight: 700;
        color: #9ca3af;
        width: 1.5rem;
    }

    .top-match-card {
        border-left: 4px solid var(--accent) !important;
        background: var(--accent-tint) !important;
    }

    .standard-code {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1f2937;
        flex: 1;
        letter-spacing: 0.01em;
    }

    .top-match-tag {
        background: var(--accent);
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.25rem 0.7rem;
        border-radius: 4px;
        text-transform: uppercase;
    }

    /* Rationale block */
    .rationale-box {
        margin-top: 2rem;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid var(--accent);
        padding: 1.5rem;
        border-radius: 4px;
    }

    .rationale-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #6b7280;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
    }

    /* Footer */
    .footer {
        background: #f3f4f6;
        color: #6b7280;
        padding: 1.2rem 2rem;
        margin-top: 3rem;
        margin-left: -5rem;
        margin-right: -5rem;
        margin-bottom: -5rem;
        border-top: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        overflow: hidden;
    }

    .footer-right {
        text-align: right;
    }
</style>

<div class="gov-bar">
    Government of India | Ministry of Consumer Affairs, Food &amp; Public Distribution | Bureau of Indian Standards
</div>

<div class="top-header">
    <div class="logo-area">
        <div class="logo-text">
            <h1>Bureau of Indian Standards</h1>
            <p>STANDARDS RECOMMENDATION ENGINE &middot; SP 21 : 2005 BUILDING MATERIALS HANDBOOK</p>
        </div>
    </div>
</div>

<div class="main-nav">
    <ul>
        <li class="active">STANDARDS SEARCH</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Initialize RAG Pipeline
@st.cache_resource
def get_pipeline():
    pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
    if not pipeline.load_existing_vectorstore():
        if os.path.exists(DATASET_PDF):
            try:
                print("Building vectorstore from PDF (first run, ~15-30s)...")
                pipeline.initialize_from_pdf(DATASET_PDF)
                print("✓ Vectorstore built successfully!")
            except Exception as e:
                print(f"Error building vectorstore: {e}")
                return None
        else:
            print("⚠ Vectorstore not found and dataset PDF missing.")
            return None
    return pipeline

try:
    pipeline = get_pipeline()
    if pipeline is None:
        st.error("❌ Failed to initialize vectorstore. Please check logs.")
except Exception as e:
    st.error(f"❌ Error initializing RAG pipeline: {e}")
    pipeline = None

# Session State for User Query
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

top_k = 5

samples = [
    "33 Grade OPC cement",
    "Portland Pozzolana Cement",
    "Hollow Concrete Blocks",
    "Ready mix concrete for RCC structures",
    "Coarse aggregates for structural concrete",
    "Common burnt clay building bricks",
    "Mild steel bars for reinforcement",
    "Toughened safety glass for windows",
    "Preservative treated timber for door frames",
    "Synthetic enamel paint for structural steel",
    "PVC pipes for cold water supply",
    "Bitumen felt for roof waterproofing",
    "Ceramic glazed tiles for flooring",
    "Rock wool thermal insulation for roofs",
]
SAMPLES_PER_ROW = 5

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="breadcrumbs">Home &gt; Standards Search &gt; <span class="current">Recommend Standards</span></div>', unsafe_allow_html=True)

st.markdown('<div class="title-section"><h2>Standards Recommendation</h2><span class="badge">AI-POWERED</span></div>', unsafe_allow_html=True)

# Dataset scope banner
st.markdown("""
<div class="scope-banner">
    <strong>📚 Knowledge Base:</strong> This engine searches <strong>BIS SP 21 : 2005 — Handbook of Building Materials</strong>.
    It covers <strong>872 Indian Standards (IS)</strong> published <strong>up to year 2005</strong>,
    across building and construction material categories.
    <div class="scope-tags">
        <span class="scope-tag">Cement</span>
        <span class="scope-tag">Concrete</span>
        <span class="scope-tag">Aggregates</span>
        <span class="scope-tag">Bricks &amp; Blocks</span>
        <span class="scope-tag">Steel</span>
        <span class="scope-tag">Glass</span>
        <span class="scope-tag">Timber</span>
        <span class="scope-tag">Paints</span>
        <span class="scope-tag">Pipes</span>
        <span class="scope-tag">Waterproofing</span>
        <span class="scope-tag">Tiles &amp; Flooring</span>
        <span class="scope-tag">Insulation</span>
    </div>
    <div style="margin-top:0.5rem; color:#666; font-size:0.76rem;">
        ⚠️ Standards published after 2005 or outside building materials are not in this database.
    </div>
</div>
""", unsafe_allow_html=True)

# Sample queries inline, wrapped across rows
st.markdown('<div class="sample-label">Try a sample query:</div>', unsafe_allow_html=True)
for row_start in range(0, len(samples), SAMPLES_PER_ROW):
    row_samples = samples[row_start:row_start + SAMPLES_PER_ROW]
    cols = st.columns(SAMPLES_PER_ROW)
    for i, sample in enumerate(row_samples):
        with cols[i]:
            if st.button(sample, key=f"sample_{sample}", use_container_width=True):
                st.session_state["query_input"] = sample
                st.rerun()

# Search Input Section
st.write("**DESCRIBE YOUR PRODUCT, MATERIAL, OR PROCESS**")
query_text = st.text_area(
    label="Query Input Box",
    label_visibility="collapsed",
    value=st.session_state["query_input"],
    placeholder="e.g. 'Supersulphated cement for marine works.'",
    height=120,
    key="main_query_area"
)

# Search Execution
if st.button("Search Standards", use_container_width=True, type="primary"):
    if not query_text.strip():
        st.warning("Please enter a product description or choose one from the sample queries.")
    elif pipeline is None:
        st.error("RAG pipeline not ready. Please verify if the database exists.")
    else:
        with st.spinner("Searching standards database..."):
            codes, rationale, latency = pipeline.process_query(query_text, top_k=top_k)

        # Display Results
        if codes:
            st.markdown(f"**{len(codes)} standards found &nbsp;·&nbsp; {latency:.2f}s latency**")

            for index, code in enumerate(codes):
                is_top = index == 0
                card_class = "standard-card top-match-card" if is_top else "standard-card"
                top_tag = '<div class="top-match-tag">Top Match</div>' if is_top else ''

                st.markdown(f"""
                <div class="{card_class}">
                    <div class="rank-badge">#{str(index+1).zfill(2)}</div>
                    <div class="standard-code">{code}</div>
                    {top_tag}
                </div>
                """, unsafe_allow_html=True)

            # Rationale Box
            st.markdown(f"""
            <div class="rationale-box">
                <div class="rationale-title">Why these?</div>
                <div style="font-size: 0.95rem; line-height: 1.6; color: #374151;">{rationale}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No standards matched the description above the confidence threshold.")

# Footer
st.markdown("""
<div class="footer">
    <div>Bureau of Indian Standards &middot; Manak Bhavan, 9 Bahadur Shah Zafar Marg, New Delhi 110002</div>
    <div class="footer-right">SP 21 : 2005 &middot; 872 Standards &middot; Hybrid BM25 + Sentence-Transformers</div>
</div>
""", unsafe_allow_html=True)
