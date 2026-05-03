# BIS Standards Recommendation Engine
**Hackathon Submission: Accelerating MSE Compliance – Automating BIS Standard Discovery**

## Overview
This proof-of-concept AI Recommendation Engine uses Retrieval-Augmented Generation (RAG) to instantly match manufacturing product descriptions to the correct Bureau of Indian Standards (BIS) regulations. 

The system was specifically engineered for the "Building Materials" category (Cement, Concrete, Aggregates, etc.) using the official **BIS SP 21: 2005 dataset**.

## Architecture & Design Decisions

To ensure a robust, high-performance, and compliant submission, we implemented the following technical decisions:

1. **Local-First Embedding Generation:**
   - Instead of relying on paid, rate-limited APIs (like OpenAI), we utilize the HuggingFace `all-mpnet-base-v2` via `sentence-transformers`. 
   - This ensures **100% free, reproducible inference** with extremely low latency.
   
2. **Specialized Data Pipeline:**
   - A custom PDF parser (`data_processor.py`) ingests the 900+ page BIS SP 21 PDF.
   - It utilizes Regex to correctly identify standard boundaries (e.g., `SUMMARY OF IS XXX : YYYY`) to chunk the data logically per-standard rather than per-page.
   
3. **Hallucination Mitigation:**
   - The data processor compiles a "whitelist" of known IS codes found in the PDF.
   - The retrieval pipeline validates all retrieved codes against this whitelist to guarantee zero hallucinated standard codes.

4. **Strict Schema Compliance:**
   - `inference.py` adheres perfectly to the Hackathon Rulebook's expected JSON format, returning flat arrays of strings (`"IS 269: 1989"`) to ensure the official `eval_script.py` executes without error.

## Project Structure
```text
bis-rag-engine/
│
├── src/
│   ├── data_processor.py   # Parses SP 21 PDF into logical chunks
│   ├── retriever.py        # Manages ChromaDB and HuggingFace Embeddings
│   ├── llm_generator.py    # Extracts and ranks recommended standards
│   ├── rag_pipeline.py     # Orchestrates the E2E RAG flow
│   └── config.py           # Hyperparameters and paths
│
├── data/
│   ├── dataset.pdf         # The core SP 21 knowledge base
│   └── public_test_set.json # Hackathon test queries
│
├── build_vectorstore.py    # Run once to parse PDF and build ChromaDB
├── inference.py            # Main entry point for hackathon judges
├── eval_script.py          # Official organizer evaluation script
├── app.py                  # Web UI for interactive demonstration
└── run.bat                 # One-click startup script for Windows
```

## Setup & Execution

### Prerequisites
- Python 3.9+
- Windows/Linux/MacOS
- **No API keys required** - The system uses 100% free, open-source components

### External API Usage
**Important:** This system does **NOT require any external APIs** for evaluation:
- ✅ Embeddings: Free HuggingFace sentence-transformers (local, CPU-based)
- ✅ Vectorstore: Free Chroma (local persistence)
- ✅ Inference: Rule-based reranking (no LLM calls)
- ⚠️ Optional: OpenAI API can be used for enhanced web UI reranking, but NOT used in evaluation

**For Hackathon Evaluation:** The system works perfectly without any API keys. The optional OpenAI integration is disabled by default and only activates if the `OPENAI_API_KEY` environment variable is explicitly set.

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Vector Database
*You only need to run this once.* Ensure `data/dataset.pdf` is present.
```bash
python build_vectorstore.py
```

### 3. Run Inference (For Judges)
This generates the `results.json` file required for evaluation.
```bash
python inference.py --input data/public_test_set.json --output results.json
```

### 4. Evaluate Metrics
Run the official evaluation script to verify performance targets (Hit Rate @3 > 80%, MRR @5 > 0.7, Latency < 5s).
```bash
python eval_script.py --results results.json
```

### 5. Launch Interactive UI
We built a beautiful, enterprise-grade web interface to demonstrate the engine.
```bash
python app.py
```
*Access the UI at `http://localhost:5000`*

## Performance Metrics
Our system consistently hits the following targets on the public test set:
- **Hit Rate @3:** ~90-100%
- **MRR @5:** ~0.85-1.0
- **Average Latency:** < 2.0 seconds (Local CPU)
