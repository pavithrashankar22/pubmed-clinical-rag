"""
ablation_eval.py
────────────────────────────────────────────────────────────────
Runs two studies:

1. EXPANDED EVALUATION — 25 clinical questions scored on
   faithfulness, answer relevancy, context recall

2. ABLATION STUDY — compares 4 retrieval strategies on 5 questions:
   - FAISS only
   - BM25 only
   - Hybrid (FAISS + BM25)
   - Hybrid + Reranker (full pipeline)

Results saved to results/ as JSON and CSV.
────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import csv
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from rag_chain import load_vectorstore, build_rag_chain, ask, hybrid_retrieve, rerank
from rank_bm25 import BM25Okapi
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
os.makedirs("results", exist_ok=True)

# ── 25 Evaluation Questions ───────────────────────────────────
eval_questions = [
    # LLMs in Clinical Decision Support
    {
        "question": "How do large language models support clinical decision making?",
        "ground_truth": "LLMs support clinical decision making by providing diagnostic assistance, treatment recommendations, and summarizing patient records.",
    },
    {
        "question": "What are the challenges of implementing LLMs in hospital settings?",
        "ground_truth": "Challenges include data privacy, integration with EHR systems, regulatory compliance, hallucination risks, and lack of clinical validation.",
    },
    {
        "question": "How accurate are LLMs for medical diagnosis assistance?",
        "ground_truth": "LLM accuracy for diagnosis varies by specialty, with some models achieving expert-level performance on standardized tests but underperforming in complex clinical cases.",
    },
    {
        "question": "What are the risks of using generative AI in clinical workflows?",
        "ground_truth": "Risks include hallucination, bias, lack of interpretability, patient safety concerns, and over-reliance by clinicians.",
    },
    {
        "question": "What evaluation frameworks exist for clinical LLM systems?",
        "ground_truth": "Evaluation frameworks include RAGAS metrics such as faithfulness, answer relevancy, and context recall, as well as clinical-specific benchmarks.",
    },
    {
        "question": "How is GPT-4 being used in healthcare settings?",
        "ground_truth": "GPT-4 is used for clinical documentation, patient triage, medical education, and decision support in various healthcare settings.",
    },
    {
        "question": "What are the ethical concerns of AI in clinical decision making?",
        "ground_truth": "Ethical concerns include algorithmic bias, lack of transparency, accountability for errors, and patient consent for AI-assisted decisions.",
    },
    {
        "question": "How do LLMs compare to traditional clinical decision support systems?",
        "ground_truth": "LLMs offer more flexible natural language interaction compared to traditional rule-based systems but may lack the reliability and auditability of established clinical tools.",
    },
    # NLP and EHRs
    {
        "question": "How is NLP used to extract information from electronic health records?",
        "ground_truth": "NLP extracts clinical entities such as diagnoses, medications and procedures from unstructured EHR text using named entity recognition models.",
    },
    {
        "question": "What are the best models for clinical named entity recognition?",
        "ground_truth": "BERT-based models fine-tuned on clinical corpora such as BioBERT and ClinicalBERT achieve state-of-the-art performance on clinical NER tasks.",
    },
    {
        "question": "How do transformer models perform on medical text classification?",
        "ground_truth": "Transformer models significantly outperform traditional ML methods on medical text classification achieving high F1 scores on clinical benchmarks.",
    },
    {
        "question": "What challenges exist in processing unstructured clinical notes?",
        "ground_truth": "Challenges include medical abbreviations, inconsistent formatting, domain-specific terminology, and patient privacy concerns.",
    },
    {
        "question": "How is BERT used for clinical NLP tasks?",
        "ground_truth": "BERT is fine-tuned on clinical corpora for tasks including named entity recognition, relation extraction, and clinical text classification.",
    },
    {
        "question": "What is the role of NLP in automated clinical coding?",
        "ground_truth": "NLP automates ICD coding by extracting diagnoses and procedures from clinical notes reducing manual coding time and errors.",
    },
    {
        "question": "How can NLP improve patient discharge summary analysis?",
        "ground_truth": "NLP can automatically extract key clinical events, medications, and follow-up instructions from discharge summaries to improve care coordination.",
    },
    {
        "question": "What datasets exist for training clinical NLP models?",
        "ground_truth": "Key datasets include MIMIC-III, i2b2, n2c2 shared task datasets, and PubMed abstracts for pre-training clinical language models.",
    },
    # Hallucination and Safety
    {
        "question": "What are the risks of hallucination in medical AI systems?",
        "ground_truth": "Hallucination in medical AI refers to generation of fabricated information which poses risks including misdiagnosis and incorrect treatment recommendations.",
    },
    {
        "question": "How does RAG reduce hallucination in medical question answering?",
        "ground_truth": "RAG reduces hallucination by grounding LLM responses in retrieved documents rather than relying on parametric knowledge.",
    },
    {
        "question": "What methods exist to detect hallucination in LLM outputs?",
        "ground_truth": "Methods include faithfulness scoring, fact verification against source documents, and consistency checking across multiple model outputs.",
    },
    {
        "question": "How can prompt engineering reduce hallucination in clinical AI?",
        "ground_truth": "Prompt engineering techniques such as chain of thought, instruction following, and explicit source citation requirements reduce hallucination rates.",
    },
    {
        "question": "What is the impact of AI hallucination on patient safety?",
        "ground_truth": "AI hallucination can lead to incorrect clinical recommendations, medication errors, and misdiagnosis posing serious patient safety risks.",
    },
    {
        "question": "How do researchers evaluate faithfulness in medical AI systems?",
        "ground_truth": "Faithfulness is evaluated by checking whether AI-generated claims are supported by retrieved source documents using automated scoring.",
    },
    {
        "question": "What are the limitations of LLMs in high-stakes medical decisions?",
        "ground_truth": "Limitations include lack of real-time data access, inability to examine patients, hallucination risk, and absence of clinical accountability.",
    },
    {
        "question": "How does retrieval augmented generation improve medical AI reliability?",
        "ground_truth": "RAG improves reliability by retrieving current peer-reviewed literature before generating answers ensuring responses are grounded in evidence.",
    },
    {
        "question": "What regulatory frameworks exist for AI safety in healthcare?",
        "ground_truth": "Regulatory frameworks include FDA guidelines for AI medical devices, EU AI Act provisions for high-risk AI, and HIPAA for data privacy.",
    },
]

# ── 5 Ablation Questions ──────────────────────────────────────
ablation_questions = [
    {
        "question": "How do LLMs support clinical decision making?",
        "ground_truth": "LLMs support clinical decision making by providing diagnostic assistance and treatment recommendations.",
    },
    {
        "question": "What are the risks of hallucination in medical AI?",
        "ground_truth": "Hallucination in medical AI poses risks including misdiagnosis and incorrect treatment recommendations.",
    },
    {
        "question": "How does RAG reduce errors in medical question answering?",
        "ground_truth": "RAG reduces errors by grounding LLM responses in retrieved documents rather than parametric knowledge.",
    },
    {
        "question": "What are challenges of deploying AI in clinical settings?",
        "ground_truth": "Challenges include data privacy, bias, interpretability, regulatory compliance, and workflow integration.",
    },
    {
        "question": "How is NLP used to extract information from EHRs?",
        "ground_truth": "NLP extracts clinical entities such as diagnoses and medications from unstructured EHR text.",
    },
]


# ── LLM Judge ─────────────────────────────────────────────────
def llm_judge(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    text = response.choices[0].message.content.strip()
    try:
        return min(max(float(text), 0.0), 1.0)
    except:
        return 0.5


def score_faithfulness(answer, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""Score whether the ANSWER is supported by the CONTEXT.
CONTEXT: {context_text}
ANSWER: {answer}
Score 1.0 = fully supported, 0.5 = partial, 0.0 = not supported.
Respond with ONLY a decimal number between 0.0 and 1.0."""
    return llm_judge(prompt)


def score_answer_relevancy(question, answer):
    prompt = f"""Score whether the ANSWER addresses the QUESTION.
QUESTION: {question}
ANSWER: {answer}
Score 1.0 = directly answers, 0.5 = partial, 0.0 = irrelevant.
Respond with ONLY a decimal number between 0.0 and 1.0."""
    return llm_judge(prompt)


def score_context_recall(question, contexts, ground_truth):
    context_text = "\n\n".join(contexts)
    prompt = f"""Score whether the CONTEXT contains enough info to answer the QUESTION given the GROUND TRUTH.
QUESTION: {question}
GROUND TRUTH: {ground_truth}
CONTEXT: {context_text}
Score 1.0 = complete, 0.5 = partial, 0.0 = missing key info.
Respond with ONLY a decimal number between 0.0 and 1.0."""
    return llm_judge(prompt)


# ── Retrieval Methods for Ablation ────────────────────────────
def retrieve_faiss_only(question, vectorstore, k=3):
    return vectorstore.similarity_search(question, k=k)


def retrieve_bm25_only(question, bm25, docs, k=3):
    tokens = question.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_idx]


def retrieve_hybrid(question, vectorstore, bm25, docs, k=15):
    return hybrid_retrieve(question, vectorstore, bm25, docs, k)


def retrieve_hybrid_rerank(question, vectorstore, bm25, docs):
    candidates = hybrid_retrieve(question, vectorstore, bm25, docs)
    return rerank(question, candidates)


# ── Run Expanded Evaluation (25 questions) ────────────────────
def run_expanded_evaluation(vectorstore, chain, bm25, docs):
    print(f"\nRunning expanded evaluation — {len(eval_questions)} questions...")
    results = []

    for i, item in enumerate(eval_questions, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(eval_questions)}] {q[:55]}...")

        result = ask((chain, vectorstore, bm25, docs), q, [])
        answer = result["answer"]
        contexts = [doc.page_content for doc in result["chunks"]]

        faith = score_faithfulness(answer, contexts)
        relev = score_answer_relevancy(q, answer)
        recall = score_context_recall(q, contexts, gt)
        overall = (faith + relev + recall) / 3

        results.append(
            {
                "question": q,
                "answer": answer,
                "faithfulness": faith,
                "answer_relevancy": relev,
                "context_recall": recall,
                "overall": overall,
            }
        )
        print(
            f"   F:{faith:.2f} R:{relev:.2f} C:{recall:.2f} O:{overall:.2f}"
        )
        time.sleep(12)

    # Averages
    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_r = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_c = sum(r["context_recall"] for r in results) / len(results)
    avg_o = (avg_f + avg_r + avg_c) / 3

    summary = {
        "total_questions": len(results),
        "avg_faithfulness": round(avg_f, 3),
        "avg_answer_relevancy": round(avg_r, 3),
        "avg_context_recall": round(avg_c, 3),
        "avg_overall": round(avg_o, 3),
        "results": results,
    }

    # Save JSON
    with open("results/evaluation_25q.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    with open("results/evaluation_25q.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "faithfulness", "answer_relevancy",
                        "context_recall", "overall"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {k: r[k] for k in ["question", "faithfulness",
                                    "answer_relevancy", "context_recall", "overall"]}
            )

    print(f"\n{'='*55}")
    print("  EXPANDED EVALUATION RESULTS (25 questions)")
    print(f"{'='*55}")
    print(f"  Faithfulness     : {avg_f:.3f}")
    print(f"  Answer Relevancy : {avg_r:.3f}")
    print(f"  Context Recall   : {avg_c:.3f}")
    print(f"  Overall Score    : {avg_o:.3f}")
    print(f"{'='*55}")
    print("  Saved: results/evaluation_25q.json")
    print("  Saved: results/evaluation_25q.csv")

    return summary


# ── Run Ablation Study (4 methods x 5 questions) ─────────────
def run_ablation_study(vectorstore, chain, bm25, docs):
    print(f"\nRunning ablation study — 4 methods x {len(ablation_questions)} questions...")

    methods = {
        "faiss_only": lambda q: retrieve_faiss_only(q, vectorstore),
        "bm25_only": lambda q: retrieve_bm25_only(q, bm25, docs),
        "hybrid": lambda q: retrieve_hybrid(q, vectorstore, bm25, docs)[:3],
        "hybrid_rerank": lambda q: retrieve_hybrid_rerank(q, vectorstore, bm25, docs),
    }

    ablation_results = {m: [] for m in methods}

    for method_name, retriever_fn in methods.items():
        print(f"\n  Method: {method_name}")
        for item in ablation_questions:
            q = item["question"]
            gt = item["ground_truth"]
            print(f"    Q: {q[:50]}...")

            retrieved_docs = retriever_fn(q)
            contexts = [doc.page_content for doc in retrieved_docs]

            # Generate answer using retrieved docs directly
            from rag_chain import format_docs_with_sources, CLINICAL_RAG_PROMPT
            from langchain_groq import ChatGroq
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatGroq(model="llama-3.1-8b-instant",
                          temperature=0.1, max_tokens=512)
            chain_eval = CLINICAL_RAG_PROMPT | llm | StrOutputParser()
            answer = chain_eval.invoke({
                "context": format_docs_with_sources(retrieved_docs),
                "chat_history": "No previous conversation.",
                "question": q,
            })

            faith = score_faithfulness(answer, contexts)
            relev = score_answer_relevancy(q, answer)
            recall = score_context_recall(q, contexts, gt)
            overall = (faith + relev + recall) / 3

            ablation_results[method_name].append({
                "question": q,
                "faithfulness": faith,
                "answer_relevancy": relev,
                "context_recall": recall,
                "overall": overall,
            })
            print(f"      F:{faith:.2f} R:{relev:.2f} C:{recall:.2f} O:{overall:.2f}")
            time.sleep(12)

    # Compute averages per method
    ablation_summary = {}
    print(f"\n{'='*60}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    print(f"  {'Method':<20} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Overall':>9}")
    print(f"  {'-'*52}")

    for method_name, results in ablation_results.items():
        avg_f = sum(r["faithfulness"] for r in results) / len(results)
        avg_r = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_c = sum(r["context_recall"] for r in results) / len(results)
        avg_o = (avg_f + avg_r + avg_c) / 3
        ablation_summary[method_name] = {
            "avg_faithfulness": round(avg_f, 3),
            "avg_answer_relevancy": round(avg_r, 3),
            "avg_context_recall": round(avg_c, 3),
            "avg_overall": round(avg_o, 3),
            "results": results,
        }
        print(f"  {method_name:<20} {avg_f:>7.3f} {avg_r:>7.3f} {avg_c:>7.3f} {avg_o:>9.3f}")

    print(f"{'='*60}")

    # Save
    with open("results/ablation_study.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)

    # Save CSV
    with open("results/ablation_study.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "avg_faithfulness", "avg_answer_relevancy",
                        "avg_context_recall", "avg_overall"],
        )
        writer.writeheader()
        for method_name, summary in ablation_summary.items():
            writer.writerow({
                "method": method_name,
                "avg_faithfulness": summary["avg_faithfulness"],
                "avg_answer_relevancy": summary["avg_answer_relevancy"],
                "avg_context_recall": summary["avg_context_recall"],
                "avg_overall": summary["avg_overall"],
            })

    print("  Saved: results/ablation_study.json")
    print("  Saved: results/ablation_study.csv")
    return ablation_summary


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found")
        exit(1)

    print("Loading RAG system...")
    vectorstore = load_vectorstore()
    chain, bm25, docs = build_rag_chain(vectorstore)

    # Run both studies
    run_expanded_evaluation(vectorstore, chain, bm25, docs)
    run_ablation_study(vectorstore, chain, bm25, docs)

    print("\nAll done! Check results/ folder.")