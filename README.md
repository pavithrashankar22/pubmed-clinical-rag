# 🏥 Clinical Q&A Bot — PubMed RAG

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers clinical questions using evidence from real PubMed research articles — with citations grounded in actual literature.

---

## Evaluation Results

| Metric | Basic RAG | Optimized RAG | Improvement |
|---|---|---|---|
| Faithfulness | 0.620 | **0.680** | +0.06 |
| Answer Relevancy | 0.840 | **0.820** | stable |
| Context Recall | 0.560 | **0.620** | +0.06 |
| **Overall** | **0.673** | **0.707** | **+0.034** |

*Evaluated using custom LLM-as-a-judge framework (Groq LLaMA-3.1)*

---

## Architecture

```
User Question + Chat History
         ↓
  Hybrid Retrieval
  FAISS (semantic) + BM25 (keyword)
  → 25-30 candidates
         ↓
  Cross-Encoder Reranker
  ms-marco-MiniLM-L-6-v2
  → Top 3 most relevant chunks
         ↓
  Prompt Template
  (context + history + question)
         ↓
  Groq LLaMA-3.1-8b
         ↓
  Grounded Answer + PubMed Citations
```

---

## Features

- Fetches real PubMed abstracts via free E-utilities API
- **228 articles** across 3 healthcare AI topics
- **2,135 searchable chunks** (300 char, 75 overlap)
- Hybrid search — FAISS semantic + BM25 keyword
- Cross-encoder reranking for precise relevance
- Conversational memory — follow-up questions work
- Custom evaluation framework (faithfulness, relevancy, recall)
- Clean Streamlit UI with sample questions

---

## Tech Stack

| Component | Tool |
|---|---|
| Data source | PubMed E-utilities API (free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| Keyword search | BM25Okapi (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG framework | LangChain |
| LLM | Groq LLaMA-3.1-8b (free tier) |
| Evaluation | Custom LLM-as-a-judge |
| UI | Streamlit |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/pavithrashankar22/pubmed-clinical-rag
cd pubmed-clinical-rag
pip install -r requirements.txt

# 2. Add your free Groq API key (console.groq.com)
cp .env.example .env

# 3. Fetch PubMed articles
python src/pubmed_fetch.py

# 4. Build the vector index
python src/build_index.py

# 5. Launch the app
streamlit run src/app.py

# 6. Run evaluation
python src/evaluate.py
```

---

## Project Structure

```
pubmed-clinical-rag/
├── src/
│   ├── pubmed_fetch.py    ← fetch + clean PubMed abstracts
│   ├── build_index.py     ← chunk + embed + FAISS index
│   ├── rag_chain.py       ← hybrid retrieval + reranking + memory
│   ├── evaluate.py        ← custom LLM-as-a-judge evaluation
│   └── app.py             ← Streamlit UI
├── data/
│   └── abstracts.json     ← 228 PubMed articles
├── .env.example
├── requirements.txt
└── README.md
```

---

## Sample Questions

- How do large language models help with clinical decision support?
- What are the risks of hallucination in medical AI systems?
- How does RAG reduce errors in medical question answering?
- How is NLP used to extract information from EHRs?

---

> ⚠️ Research and portfolio project only. Not intended for clinical use.

*Built by Pavithra Shankar — AI/ML Engineer*
