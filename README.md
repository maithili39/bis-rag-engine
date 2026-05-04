# BIS Standards Recommendation Engine
**Hackathon Submission: Accelerating MSE Compliance – Automating BIS Standard Discovery**

## Overview
This AI Recommendation Engine uses Retrieval-Augmented Generation (RAG) to instantly match manufacturing product descriptions to the correct Bureau of Indian Standards (BIS) regulations.

Built specifically for the **Building Materials** category (Cement, Concrete, Aggregates, etc.) using the official **BIS SP 21: 2005 dataset**.

---

## Architecture & Design Decisions

1. **Local-First Embedding Generation**
   - Uses HuggingFace `all-mpnet-base-v2` via `sentence-transformers` — free, local, no API key needed.
   - 768-dimensional embeddings optimized for information retrieval (beats general-purpose models).
   - Ensures 100% reproducible inference on normal hardware.

2. **Hybrid Retrieval (Semantic + BM25)**
   - Combines vector similarity search (ChromaDB) with BM25 keyword matching.
   - Weighted 40% semantic / 60% keyword — domain-specific technical terminology (IS codes, standards) matching is critical.

3. **Hallucination Mitigation**
   - A whitelist of all IS codes extracted from SP 21 is used to validate every retrieved code.
   - Guarantees zero hallucinated standard codes in output.

4. **Rationale Generation**
   - Each recommendation includes a plain-language explanation of why the standard was matched.

5. **Strict Schema Compliance**
   - `inference.py` outputs exactly the format expected by `eval_script.py`.

---

## Project Structure

```text
bis-rag-engine/
│
├── src/
│   ├── data_processor.py   # Parses SP 21 PDF into per-standard chunks
│   ├── retriever.py        # ChromaDB + HuggingFace embeddings + BM25
│   ├── llm_generator.py    # Reranking, rationale generation
│   ├── rag_pipeline.py     # End-to-end RAG orchestration
│   └── config.py           # Hyperparameters and paths
│
├── data/
│   ├── dataset.pdf         # BIS SP 21: 2005 knowledge base (place here)
│   └── public_test_set.json
│
├── build_vectorstore.py    # Run once — builds ChromaDB from the PDF
├── inference.py            # ⭐ Main entry point for judges
├── eval_script.py          # Official organizer evaluation script (unmodified)
├── streamlit_app.py        # Interactive web UI
├── requirements.txt
└── README.md
```

---

## Demo Video
🎥 **[Watch the Project Demo Video Here](https://drive.google.com/drive/folders/1giZQhaWp3LL3_M2CiFrCCX46HhmU38i5)**

---

## Setup & Execution

### System Requirements & Prerequisites
- **OS**: Windows / Linux / macOS
- **Python**: 3.9 or higher
- **RAM**: Minimum 4GB RAM required.
- **Hardware**: **GPU Highly Recommended/Required**. While the system can run on a CPU, the initial model loading and vectorstore build can take up to **10 minutes** without a dedicated GPU.
- **API Keys**: **None required** — uses 100% free, local, open-source components.

### Optional: OpenAI API Key
If you want LLM-enhanced rationale generation in the UI, set:
```bash
# Windows
set OPENAI_API_KEY=sk-...

# Linux/macOS
export OPENAI_API_KEY=sk-...
```
The system works perfectly **without** this. Rule-based rationale is used by default.

---

### Step 1 — Clone the Repository
```bash
git clone https://github.com/maithili39/bis-rag-engine
cd bis-rag-engine
```

### Step 2 — Set Up a Virtual Environment (Recommended)
Creating a virtual environment ensures a clean workspace and prevents dependency conflicts.
```bash
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ If you see a `sentence-transformers` import error, ensure you're on the correct version:
> `pip install sentence-transformers==2.7.0 transformers==4.41.2`

### Step 4 — Place the Dataset
Copy the official BIS SP 21 PDF into the `data/` folder. Ensure the file is named exactly:
```text
data/dataset.pdf
```

### Step 5 — Build the Vector Database (Run Once)
This step parses the PDF and builds the ChromaDB index. **This step may take up to 10 minutes on a CPU.**
```bash
python build_vectorstore.py
```
*Expected output: `BUILD COMPLETE` along with the total document count.*

### Step 6 — Run Inference (For Evaluation)
This runs the full RAG pipeline against the test dataset and generates standard recommendations.
```bash
python inference.py --input data/public_test_set.json --output results.json
```

> **Note on Initial Latency & Hardware**: The first time the application starts (or when running `build_vectorstore.py` / `inference.py`), the heavy local embedding model is loaded into memory. **Without a GPU, this initial cold start can take up to 10 minutes.** Additionally, the **very first search query** may take slightly longer (a few seconds) as the system warms up and allocates memory for inference. However, once fully warmed up, subsequent queries take only a fraction of a second. The `eval_script.py` calculates the *average* latency across the entire test set, so these one-time loading delays are amortized and do not penalize the overall performance score.

### Step 7 — Evaluate Metrics
Check your system's performance against the official hackathon targets.
```bash
python eval_script.py --results results.json
```
*Expected output: Your Hit Rate @3, MRR @5, and Average Latency scores.*

### Step 8 — Launch the Web UI
Run the interactive Streamlit application to visually test queries.
```bash
streamlit run streamlit_app.py
```
The app will automatically open in your browser at **http://localhost:8501**.
*(To stop the server, press `Ctrl+C` in your terminal).*

---

## Input / Output Format

### Input JSON (provided by judges)
```json
[
  {
    "id": "PUB-01",
    "query": "We manufacture 33 Grade Ordinary Portland Cement...",
    "expected_standards": ["IS 269: 1989"]
  }
]
```
> `expected_standards` may or may not be present in the private test dataset.

### Output JSON (written to `--output` file)
```json
[
  {
    "id": "PUB-01",
    "query": "We manufacture 33 Grade Ordinary Portland Cement...",
    "retrieved_standards": ["IS 269: 1989", "IS 8112: 1989", "IS 12269: 1987", "IS 455: 1989", "IS 1489 (Part 1): 1991"],
    "rationale": "IS 269: 1989 is the primary applicable standard, covering ordinary portland cement specifications.",
    "latency_seconds": 1.12,
    "expected_standards": ["IS 269: 1989"]
  }
]
```
> `expected_standards` is only echoed in output when present in the input, so `eval_script.py` can score correctly.

---

## Performance Targets

| Metric | Target | Our Score (Public Set) |
|---|---|---|
| Hit Rate @3 | > 80% | 100% |
| MRR @5 | > 0.7 | 0.9500 |
| Avg Latency | < 5.0s | < 2.0s |

---

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` (all-mpnet-base-v2, 768-dim) |
| Vector Store | `ChromaDB` |
| Keyword Search | `rank-bm25` |
| PDF Parsing | `pypdf` |
| LLM (optional) | `langchain-openai` (gpt-3.5-turbo) |
| Web UI | `Streamlit` |
