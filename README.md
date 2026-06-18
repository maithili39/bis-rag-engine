---
title: BIS Standards Finder
colorFrom: green
colorTo: orange
sdk: streamlit
sdk_version: "1.45.0"
app_file: streamlit_app.py
pinned: true
---

# BIS Standards Recommendation Engine

A RAG (Retrieval-Augmented Generation) system that maps manufacturing product descriptions to the correct **Bureau of Indian Standards (BIS)** codes. Built for the **BIS × Startup Studio Hackathon 2026**.

> **Live demo:** [bisstandardrecomender.streamlit.app](https://bisstandardrecomender.streamlit.app/)

---

## ⚠️ Scope & Limitations

| What it covers | What it does NOT cover |
|---|---|
| BIS SP 21 : 2005 — Building Materials Handbook | Standards published after 2005 |
| **872 IS standards** across cement, concrete, aggregates, bricks, steel, glass, timber, pipes, paints, tiles, waterproofing | Electrical, mechanical, food, or textile standards |
| Natural-language queries in English | Ambiguous or out-of-domain queries |

**This engine only knows standards up to the year 2005.** It cannot recommend newer amendments or revisions.

---

## Performance

| Metric | Score | Target |
|---|---|---|
| Hit Rate @ K=3 | **100%** | > 80% |
| MRR @ K=5 | **1.000** | > 0.70 |
| Avg Latency | **1.16 s** | < 5 s |

Evaluated on the official 10-query public test set. Small sample — real-world numbers on diverse queries will differ.

---

## Architecture

```
Product description
        │
        ▼
  Hybrid Retrieval
  ├─ Dense: all-mpnet-base-v2 (40%)
  └─ BM25 keyword (60%)
        │ top-12 candidates
        ▼
  Hallucination Filter
  (IS code whitelist from PDF)
        │ validated codes only
        ▼
  Keyword Reranker + Rationale
        │
        ▼
  Top-5 BIS Standards
```

**Key choices:**
- **NumPy vector index** — no C++ build tools, fast for this corpus size
- **60/40 BM25/dense split** — BM25 dominates because IS codes are exact-match keywords
- **Whitelist validation** — `known_codes.json` is auto-generated from the PDF; no hallucinated codes possible

---

## Quickstart

```bash
git clone https://github.com/maithili39/bis-rag-engine
cd bis-rag-engine
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# One-time vectorstore build (~15-30 min on CPU)
python build_vectorstore.py

# Run Streamlit UI
python -m streamlit run streamlit_app.py
```

Evaluate on the test set:
```bash
python inference.py --input data/public_test_set.json --output results.json
python eval_script.py --results results.json
```

No API key required. Set `OPENAI_API_KEY` only if you want AI-generated rationale text.

---

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` / `all-mpnet-base-v2` |
| Keyword search | `rank-bm25` |
| Vector index | NumPy |
| PDF parsing | `pypdf` |
| UI | Streamlit |
| REST API | FastAPI + Uvicorn |

---

## Project Structure

```
bis-rag-engine/
├── src/
│   ├── config.py           # Hyperparameters
│   ├── data_processor.py   # PDF parsing & chunking
│   ├── retriever.py        # Hybrid dense + BM25 retrieval
│   ├── llm_generator.py    # Reranking & rationale
│   └── rag_pipeline.py     # End-to-end orchestration
├── data/
│   ├── dataset.pdf         # BIS SP 21: 2005 source
│   └── public_test_set.json
├── tests/                  # 29 pytest tests
├── build_vectorstore.py    # One-time setup
├── streamlit_app.py        # Web UI
├── inference.py            # Batch inference
└── eval_script.py          # Official evaluation script
```

---

*Built for BIS × Startup Studio Hackathon 2026 · [github.com/maithili39/bis-rag-engine](https://github.com/maithili39/bis-rag-engine)*
