# 🏥 Clinical Q&A Bot — PubMed RAG

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers 
clinical questions using evidence from real PubMed research articles — 
with citations grounded in actual literature.

---

## How It Works
User Question
↓
Embed Question (HuggingFace sentence-transformers)
↓
FAISS Vector Search → Top 3 Relevant PubMed Chunks
↓
Prompt Template (context + question injected)
↓
Groq LLaMA → Grounded Answer + PubMed Citations

---

## Tech Stack

| Component | Tool |
|---|---|
| Data source | PubMed E-utilities API (free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| RAG framework | LangChain |
| LLM | Groq LLaMA-3.1-8b |
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
```

---

## Project Structure
pubmed-clinical-rag/
├── src/
│   ├── pubmed_fetch.py    ← fetch + clean PubMed abstracts
│   ├── build_index.py     ← chunk + embed + FAISS index
│   ├── rag_chain.py       ← retrieval + LLM chain
│   └── app.py             ← Streamlit UI
├── data/
│   └── abstracts.json     ← 79 fetched PubMed articles
├── .env.example
├── requirements.txt
└── README.md

---

## Sample Questions
- How do large language models help with clinical decision support?
- What are the risks of hallucination in medical AI systems?
- How does RAG reduce errors in medical question answering?

---

> ⚠️ Research and portfolio project only. Not intended for clinical use.

*Built by Pavithra Shankar — AI/ML Engineer*
