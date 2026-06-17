---
title: BIS Standards Finder
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: true
---

# BIS Standards Recommendation Engine

A RAG (Retrieval-Augmented Generation) system that maps manufacturing product descriptions to the correct Bureau of Indian Standards (BIS) regulations. Built for the **BIS x Startup Studio Hackathon 2026** — "Accelerating MSE Compliance".

**🚀 [Try it live on HuggingFace Spaces](https://maithili39-bis-rag-engine.hf.space/)**

---

## Performance

| Metric | Score | Hackathon Target |
|---|---|---|
| Hit Rate @ K=3 | **100%** | > 80% |
| MRR @ K=5 | **1.0000** | > 0.70 |
| Avg Latency | **1.16 s** | < 5 s |
| Max Latency | **1.30 s** | < 5 s |

Evaluated on the public 10-query test set using the official `eval_script.py`.

**Is this good?** Yes — 100% Hit@3 means the correct standard appears in every top-3 result, and MRR of 1.0 means the correct answer is ranked first every time on this test set. The 1.16s average latency is well under the 5s target.

**Honest caveat:** The public test set has only 10 queries, which is a small sample. Real-world performance on a larger or more ambiguous query set would likely show lower numbers. The system is optimised for the Building Materials category of BIS SP 21: 2005 and may not generalise well to other domains or newer standards.

---

## What this project does well

- **Hybrid retrieval that matches the domain.** BIS standards use highly specific technical codes ("IS 269: 1989"). BM25 keyword matching (60%) handles exact code/term lookups; dense embeddings (40%) handle paraphrased natural-language queries. Tuning this split to 40/60 improved results over semantic-only.
- **Zero hallucinations by design.** Every retrieved code is validated against a whitelist of known IS codes extracted from the source PDF. Fabricated codes are impossible in the output.
- **No external API needed.** The full pipeline runs locally with free HuggingFace models. No OpenAI key, no cloud calls, no cost at inference time.
- **No compiler dependencies.** Replaced ChromaDB (which requires C++ build tools) with a NumPy in-memory vector index — a single matrix-vector dot product for cosine similarity. Same accuracy, installs anywhere, deploys on HF Spaces without issues.
- **Clean code with tests.** 29 pytest tests cover parsing, hallucination filtering, schema compliance, and metric sanity checks. CI runs on GitHub Actions.

---

## What could be improved

- **Small evaluation set.** 10 queries is not enough to draw strong conclusions. A proper evaluation would use 100+ diverse queries, including ambiguous ones that don't directly name the standard.
- **Cross-encoder reranking not implemented.** A cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) would improve ranking quality by scoring query-document pairs jointly rather than independently. This is the standard next step in RAG pipelines.
- **PDF parsing is fragile.** The SP 21 PDF has a malformed internal structure. pypdf emits 6000+ warnings and some standards may be missed or split incorrectly. A cleaner data source or a dedicated table-of-contents parser would help.
- **Rule-based rationale is weak.** The "Why these?" explanation uses keyword matching against a hardcoded dictionary of ~15 standards. It breaks silently for any standard not in that dictionary.
- **Coverage is narrow.** Only Building Materials (cement, concrete, aggregates) from SP 21: 2005. Expanding to other BIS categories or newer standards would make it genuinely production-useful.

---

## Architecture

```
Product description (natural language)
         |
         v
+---------------------------------------+
|          Hybrid Retrieval             |
|   Dense embeddings    BM25 keyword    |
|   (all-mpnet-base-v2) (rank-bm25)     |
|      40% weight          60% weight   |
|              |                        |
|         Combined score                |
+---------------|-----------------------+
                | top-12 candidates
                v
    +---------------------+
    |  Hallucination filter|
    |  (IS code whitelist) |
    +----------+----------+
               | validated codes only
               v
    +---------------------+
    |  Keyword reranker   |
    |  + rationale text   |
    +----------+----------+
               |
               v
       Top-5 BIS standards
```

Key design choices:
- **NumPy vector index** instead of a vector database — the corpus (~500 chunks) is small enough that `matrix @ query_vector` (cosine similarity) is faster and has no native dependencies.
- **Two-stage chunking** — each standard is stored as both a full document and overlapping sub-chunks for better recall on long standards.
- **Whitelist validation** — `known_codes.json` is generated from the PDF itself during the build step, so it always matches the actual source data.

---

## Tech stack

| Component | Library | Reason |
|---|---|---|
| Embeddings | `sentence-transformers` / `all-mpnet-base-v2` | Free, local, 768-dim, strong retrieval quality |
| Keyword search | `rank-bm25` | Critical for exact IS code and technical term matching |
| Vector index | NumPy | No compiler needed, fast enough for this corpus size |
| PDF parsing | `pypdf` | Pure Python, no system dependencies |
| LLM reranking (optional) | OpenAI `gpt-4o-mini` | Only for the UI "Why these?" explanation — not used in inference |
| REST API | FastAPI + Uvicorn | Serves `/query`, `/health`, `/standards` endpoints + static frontend |
| Web UI | Vanilla HTML/CSS/JS (served by FastAPI) | HF Spaces entry point uses Streamlit (`app.py`) |

---

## Project structure

```
bis-rag-engine/
├── src/
│   ├── config.py           # All hyperparameters
│   ├── data_processor.py   # PDF parsing and chunking
│   ├── retriever.py        # NumPy dense + BM25 hybrid retrieval
│   ├── llm_generator.py    # Code extraction, filtering, reranking, rationale
│   └── rag_pipeline.py     # End-to-end orchestration
├── tests/                  # 29 pytest tests
├── .github/workflows/      # CI (lint + tests on push)
├── data/
│   ├── dataset.pdf         # BIS SP 21: 2005 source document
│   └── public_test_set.json
├── build_vectorstore.py    # One-time setup: PDF -> embeddings
├── inference.py            # Batch inference entry point
├── eval_script.py          # Official evaluation script (unmodified)
├── run_extended_eval.py    # Extended A/B evaluation runner
├── api.py                  # FastAPI server + serves frontend/
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/maithili39/bis-rag-engine
cd bis-rag-engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_vectorstore.py      # one-time, ~10-30 min on CPU
```

Run inference and evaluate:
```bash
python inference.py --input data/public_test_set.json --output results.json
python eval_script.py --results results.json
```

Run the web UI:
```bash
streamlit run app.py
```

Run tests:
```bash
pytest tests/ -v
```

No API key required. Set `OPENAI_API_KEY` only if you want AI-generated rationale text in the UI — the inference pipeline never calls OpenAI.

---

## Configuration

All hyperparameters are in `src/config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | HuggingFace model for dense embeddings |
| `SEMANTIC_WEIGHT` | `0.4` | Weight of dense similarity in hybrid score |
| `BM25_WEIGHT` | `0.6` | Weight of BM25 keyword score |
| `TOP_K_RETRIEVAL` | `12` | Candidates fetched before reranking |
| `TOP_K_RESULTS` | `5` | Final standards returned per query |
| `MIN_CONFIDENCE_SCORE` | `0.30` | Minimum hybrid score to include a result (tune down for recall, up for precision) |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `250` | Overlap between adjacent chunks |

---

## Usage Examples & Case Studies

### **Case Study 1: Cement Manufacturer**

**Scenario:** A small cement plant needs to ensure compliance with Indian standards.

**Query:** `"We are manufacturing 33-Grade Ordinary Portland Cement. What standard applies?"`

**System Response:**
```
IS 269: 1989 — Ordinary Portland Cement (OPC)
├─ Confidence: High
├─ Covers: Chemical, physical requirements, testing methods
├─ Best for: General-purpose concrete
└─ Action: Implement quality checks for fineness, setting time, strength
```

**Impact:** The manufacturer can immediately reference the correct standard for product specification and quality assurance.

---

### **Case Study 2: Construction Aggregate Supplier**

**Scenario:** A quarry operator needs to supply aggregates for a concrete project.

**Query:** `"We extract natural sand and gravel. What standards govern their quality for concrete use?"`

**System Response:**
```
IS 383: 1970 — Coarse and Fine Aggregates
├─ Covers: Gradation, particle size, impurities
├─ Applies to: Structural concrete
└─ Testing: Sieve analysis, crushing value, water absorption
```

**Impact:** The supplier knows exactly which tests to perform and can certificate compliance to buyers.

---

### **Case Study 3: Concrete Masonry Manufacturer**

**Scenario:** A company transitions from traditional bricks to hollow lightweight concrete blocks.

**Query:** `"We manufacture hollow and solid lightweight concrete masonry blocks. What are the dimension and physical requirements?"`

**System Response:**
```
IS 2185 (Part 2): 1983 — Hollow and Solid Lightweight Concrete Masonry Units
├─ Covers: Dimensions, strength, density
├─ Applications: Interior/exterior walls, non-load-bearing
└─ Certification: Compression strength ≥ 3.5 MPa
```

**Impact:** Clear manufacturing targets and test protocols for production quality control.

---

### **What This System Does *Well***

✅ **Exact matches:** "IS 269" → Finds IS 269: 1989 instantly  
✅ **Paraphrased queries:** "Portland slag cement composition" → Finds IS 455: 1989  
✅ **Multi-part matches:** "Aggregates for concrete" → Returns both IS 383 and IS 2386  
✅ **No hallucinations:** Only returns standards from the source PDF

### **What This System Does *Not* Handle**

❌ **Steel standards** — Not in the dataset (Building Materials only)  
❌ **Electrical standards** — Out of scope  
❌ **Newer standards** — Limited to SP 21: 2005 data  
❌ **Ambiguous/vague queries** — "What standard should I use?" → Returns nothing  
❌ **Cross-domain lookups** — Can't recommend standards outside the dataset

---

*Built for the BIS x Startup Studio Hackathon 2026. Source: [github.com/maithili39/bis-rag-engine](https://github.com/maithili39/bis-rag-engine)*
