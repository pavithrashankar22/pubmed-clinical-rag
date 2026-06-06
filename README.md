Clinical Q&A Bot — PubMed RAG
An end-to-end Retrieval-Augmented Generation (RAG) system that answers
clinical questions using evidence from real PubMed research articles —
with citations grounded in actual peer-reviewed literature.
Built with LangChain, FAISS, BM25, cross-encoder reranking, conversational
memory, and a custom LLM-as-a-judge evaluation framework.

Demo
Show Image
Answering a clinical NLP question with real PubMed citations
Show Image
Retrieved chunks showing the source abstracts used to generate the answer
Show Image
Conversational memory — follow-up questions reference previous context

Evaluation Results
Expanded Evaluation (25 questions)
MetricScoreFaithfulness0.680Answer Relevancy0.820Context Recall0.620Overall0.707
Evaluated using custom LLM-as-a-judge framework (Groq LLaMA-3.1)

Ablation Study — Retrieval Method Comparison
MethodFaithfulnessRelevancyRecallOverallFAISS only0.8000.8000.7400.780BM25 only0.7800.6800.5000.653Hybrid0.8200.8000.7400.787Hybrid + Rerank0.8000.8000.6800.760
Key finding: Hybrid search (FAISS + BM25) outperforms all other methods.
BM25 alone shows the lowest context recall (0.500) — pure keyword search
misses the semantic nature of clinical questions.
Full results in results/ablation_study.json and results/evaluation_25q.json.

Sample Output
Question: What is the role of NLP in automated clinical coding?
Answer:

NLP plays a crucial role in automated clinical coding by offering a scalable
approach for extracting structured diagnosis codes directly from clinical text,
such as discharge summaries (Source 1: PMID 42175002, 2026). This can help
reduce label noise and improve the accuracy of ICD-10 diagnosis codes.

Sources Retrieved:

PubMed PMID 42175002 (2026): Evidence-Grounded LLM Validation of MIMIC-IV ICD Labels
PubMed PMID 42158042 (2026): Automated Identification of Cancer-Associated Thrombosis Events via NLP
PubMed PMID 42232873 (2026): Leveraging NLP for automated data extraction from cardiovascular imaging reports


Architecture
User Question + Chat History
         ↓
  Hybrid Retrieval
  ├── FAISS semantic search  (finds by meaning)
  └── BM25 keyword search    (finds by exact terms)
  → 25-30 candidates merged
         ↓
  Cross-Encoder Reranker
  ms-marco-MiniLM-L-6-v2
  → Top 3 most relevant chunks
         ↓
  Prompt Template
  (context + chat history + question)
         ↓
  Groq LLaMA-3.1-8b
         ↓
  Grounded Answer + PubMed Citations

Features

Fetches real PubMed abstracts via free NCBI E-utilities API
228 articles across 3 healthcare AI search topics
2,135 searchable chunks (300 chars, 75 char overlap)
Hybrid search — FAISS semantic + BM25 keyword retrieval
Cross-encoder reranking for precise relevance scoring
Conversational memory — follow-up questions work naturally
Custom LLM-as-a-judge evaluation (faithfulness, relevancy, recall)
Ablation study comparing 4 retrieval strategies
Clean Streamlit UI with sample questions sidebar
Code formatted with Black


Tech Stack
ComponentToolData sourcePubMed NCBI E-utilities API (free)Embeddingssentence-transformers/all-MiniLM-L6-v2Vector storeFAISSKeyword searchBM25Okapi (rank-bm25)Rerankercross-encoder/ms-marco-MiniLM-L-6-v2RAG frameworkLangChainLLMGroq LLaMA-3.1-8b (free tier)EvaluationCustom LLM-as-a-judgeUIStreamlitCode formattingBlack

Quickstart
bash# 1. Clone and install
git clone https://github.com/pavithrashankar22/pubmed-clinical-rag
cd pubmed-clinical-rag
pip install -r requirements.txt

# 2. Add your free Groq API key (console.groq.com)
cp .env.example .env
# Edit .env and add GROQ_API_KEY

# 3. Fetch PubMed articles
python src/pubmed_fetch.py

# 4. Build the vector index
python src/build_index.py

# 5. Launch the app
streamlit run src/app.py

# 6. Run full evaluation + ablation study (optional — 30-40 mins)
python src/ablation_eval.py

Project Structure
pubmed-clinical-rag/
├── src/
│   ├── pubmed_fetch.py      ← fetch + clean PubMed abstracts
│   ├── build_index.py       ← chunk + embed + FAISS index
│   ├── rag_chain.py         ← hybrid retrieval + reranking + memory
│   ├── evaluate.py          ← 5-question evaluation
│   ├── ablation_eval.py     ← 25-question eval + ablation study
│   └── app.py               ← Streamlit UI
├── data/
│   └── abstracts.json       ← 228 PubMed articles
├── results/
│   ├── evaluation_25q.json  ← full 25-question evaluation results
│   ├── evaluation_25q.csv
│   ├── ablation_study.json  ← ablation study results
│   └── ablation_study.csv
├── screenshots/
│   ├── demo_answer.png      ← answer with citations
│   ├── raw_chunks.png       ← retrieved source chunks
│   └── memory_demo.png      ← conversational memory demo
├── .env.example
├── requirements.txt
└── README.md

Sample Questions to Try

How do large language models help with clinical decision support?
What are the risks of hallucination in medical AI systems?
How does RAG reduce errors in medical question answering?
What is the role of NLP in automated clinical coding?
How is BERT used for clinical NLP tasks?
What are the challenges of deploying AI in clinical settings?
How does label noise affect model performance in clinical NLP?


Limitations

Abstracts only — searches PubMed abstracts, not full paper text. Some answers may lack depth due to abstract-level detail.
Static corpus — 228 articles fetched at build time. Re-run pubmed_fetch.py and build_index.py to incorporate new publications.
Context recall ceiling — current recall score of 0.620 reflects corpus size. Expanding to 500+ articles would improve retrieval coverage.
LLM variability — Groq LLaMA-3.1 may occasionally drift from retrieved context despite prompt constraints. Faithfulness score of 0.680 reflects this.
No full-text access — PubMed E-utilities provides abstracts only. Full-text retrieval would require PMC Open Access API integration.
Not for clinical use — research and portfolio demonstration only. Not validated for medical decision making.



⚠️ Disclaimer: This system is built for research and demonstration purposes only.
It is not validated for clinical use. Always consult licensed healthcare professionals
for medical decisions.


Built by Pavithra Shankar — AI/ML Engineer 
