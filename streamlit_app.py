import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import CHROMA_PERSIST_DIR, DATASET_PDF

# Page Configuration
st.set_page_config(
    page_title="BIS Standards Recommendation",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Forest Green & Orange Theme)
st.markdown("""
<style>
    /* Hide Default Streamlit Style Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Theme Colors Variables */
    :root {
        --bg-header: #05261e;
        --bg-nav: #093c31;
        --gov-bg: #031511;
        --accent: #e65c00;
        --accent-hover: #ff6b0a;
        --accent-light: #fff2ea;
        --border-color: #e5e5e5;
        --white: #ffffff;
    }

    /* Top Gov Bar */
    .gov-bar {
        background-color: var(--gov-bg);
        color: #a3b8b4;
        padding: 0.5rem 2rem;
        font-size: 0.75rem;
        font-weight: 500;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: -6rem;
        margin-bottom: 2rem;
        margin-left: -5rem;
        margin-right: -5rem;
    }

    /* Main Container Padding Adjustment */
    .block-container {
        padding-top: 5rem !important;
    }

    /* Header Banner */
    .top-header {
        background-color: var(--bg-header);
        color: var(--white);
        padding: 1.5rem 2rem;
        border-bottom: 2px solid var(--accent);
        margin-left: -5rem;
        margin-right: -5rem;
        margin-top: -2.1rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo-area {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-text h1 {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        color: white;
    }
    
    .logo-text p {
        font-size: 0.7rem;
        color: #a3b8b4;
        text-transform: uppercase;
        margin: 0;
        letter-spacing: 0.05em;
    }

    .portal-btn {
        background: transparent;
        color: #dddddd;
        border: 1px solid #2e5c51;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
    }
    
    /* Navigation Bar */
    .main-nav {
        background-color: var(--bg-nav);
        border-bottom: 3px solid var(--accent);
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
        padding: 0.8rem 1.5rem;
        color: #a3b8b4;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .main-nav li.active {
        color: var(--white);
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Breadcrumbs */
    .breadcrumbs {
        font-size: 0.85rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    
    .breadcrumbs .current {
        color: var(--bg-nav);
        font-weight: 500;
    }

    /* Title & Badge */
    .title-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .title-section h2 {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--bg-header);
        margin: 0;
    }
    
    .badge {
        background-color: var(--accent);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }

    /* Sidebar Settings Header */
    .sidebar .sidebar-content {
        background-color: var(--white) !important;
    }
    
    .sidebar-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--accent);
        border-left: 3px solid var(--accent);
        padding-left: 0.6rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        letter-spacing: 0.05em;
    }

    /* Card Styling */
    .standard-card {
        background: var(--white);
        border: 1px solid var(--border-color);
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        transition: transform 0.2s;
    }
    
    .standard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        border-color: #c4d4d1;
    }
    
    .rank-badge {
        font-size: 0.85rem;
        font-weight: 700;
        color: #666666;
        width: 1.5rem;
    }
    
    .top-match-card {
        border-left: 4px solid var(--bg-nav) !important;
    }
    
    .standard-code {
        font-size: 1.1rem;
        font-weight: 600;
        color: #222222;
        flex: 1;
    }
    
    .top-match-tag {
        background-color: #e0f0ed;
        color: var(--bg-nav);
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        text-transform: uppercase;
    }

    /* Rationale block */
    .rationale-box {
        margin-top: 2rem;
        background: var(--white);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--bg-nav);
        padding: 1.5rem;
        border-radius: 8px;
    }
    
    .rationale-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--bg-nav);
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
    }

    .footer {
        background-color: var(--bg-header);
        color: #7b9e97;
        padding: 1.5rem 2rem;
        margin-top: 5rem;
        margin-left: -5rem;
        margin-right: -5rem;
        margin-bottom: -5rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
    }
    
    .footer-right {
        text-align: right;
    }
</style>

<div class="gov-bar">
    Government of India | Ministry of Consumer Affairs, Food & Public Distribution | Bureau of Indian Standards
</div>

<div class="top-header">
    <div class="logo-area">
        <div class="logo-text">
            <h1>Bureau of Indian Standards</h1>
            <p>STANDARDS RECOMMENDATION ENGINE &middot; SP 21 : 2005 BUILDING MATERIALS HANDBOOK</p>
        </div>
    </div>
    <div class="portal-btn">IS Portal v2.0</div>
</div>

<div class="main-nav">
    <ul>
        <li class="active">STANDARDS SEARCH</li>
        <li>BROWSE CATALOGUE</li>
        <li>IS AMENDMENTS</li>
        <li>CERTIFICATION</li>
        <li>HELP & DOCS</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Initialize RAG Pipeline
@st.cache_resource
def get_pipeline():
    pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
    if not pipeline.load_existing_vectorstore():
        if os.path.exists(DATASET_PDF):
            pipeline.initialize_from_pdf(DATASET_PDF)
        else:
            st.error("Vectorstore not found and dataset PDF missing.")
    return pipeline

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Error initializing RAG pipeline: {e}")
    pipeline = None

# Session State for User Query
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

# Sidebar Layout
st.sidebar.markdown('<div class="sidebar-section-title">SEARCH SETTINGS</div>', unsafe_allow_html=True)
top_k = st.sidebar.slider("NUMBER OF RESULTS", min_value=1, max_value=10, value=5)

st.sidebar.markdown('<div class="sidebar-section-title">SAMPLE QUERIES</div>', unsafe_allow_html=True)

samples = [
    "33 Grade OPC",
    "Hollow Concrete Blocks",
    "Portland Pozzolana Cement",
    "White Portland Cement",
    "Reinforced Concrete"
]

for sample in samples:
    if st.sidebar.button(sample, key=sample, use_container_width=True):
        st.session_state["query_input"] = sample
        st.rerun()

# Main Container Area
st.markdown('<div class="breadcrumbs">Home &gt; Standards Search &gt; <span class="current">Recommend Standards</span></div>', unsafe_allow_html=True)

st.markdown('<div class="title-section"><h2>Standards Recommendation</h2><span class="badge">AI-POWERED</span></div>', unsafe_allow_html=True)

# Search Input Section
st.write("**DESCRIBE YOUR PRODUCT, MATERIAL, OR PROCESS**")
query_text = st.text_area(
    label="Query Input Box",
    label_visibility="collapsed",
    value=st.session_state["query_input"],
    placeholder="e.g. 'Supersulphated cement for marine works.'",
    height=120
)

# Search Execution
if st.button("Search Standards", use_container_width=True, type="primary"):
    if not query_text.strip():
        st.warning("Please enter a product description or choose one from the sample queries.")
    elif pipeline is None:
        st.error("RAG pipeline not ready. Please verify if the database exists.")
    else:
        with st.spinner("Searching standards database..."):
            start_time = time.time()
            codes, rationale, latency = pipeline.process_query(query_text, top_k=top_k)
            
        # Display Results
        if codes:
            st.markdown(f"**{len(codes)} standards found • {latency:.2f}s latency**")
            
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
                <div style="font-size: 0.95rem; line-height: 1.6; color: #222222;">{rationale}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No standards matched the description above the confidence threshold.")

# Footer section
st.markdown("""
<div class="footer">
    <div>Bureau of Indian Standards &middot; Manak Bhavan, 9 Bahadur Shah Zafar Marg, New Delhi 110002</div>
    <div class="footer-right">BIS knowledge base loads on demand &middot; Hybrid retrieval: BM25 + TF-IDF + Sentence-Transformers</div>
</div>
""", unsafe_allow_html=True)
