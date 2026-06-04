"""
evaluate.py
────────────────────────────────────────────────────────────────
Custom RAG Evaluation — measures 3 metrics using Groq as judge:

  Faithfulness    — is the answer grounded in retrieved context?
  Answer Relevancy — does the answer address the question?
  Context Recall  — did retriever find relevant chunks?

Why custom instead of RAGAS?
  More control, no dependency issues, same research validity.
  Uses LLM-as-a-judge pattern — standard in AI evaluation.
────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from rag_chain import load_vectorstore, build_rag_chain, ask
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Evaluation questions + ground truths ──────────────────────
eval_questions = [
    {
        "question": "How do large language models support clinical decision making?",
        "ground_truth": "LLMs support clinical decision making by providing diagnostic assistance, treatment recommendations, and summarizing patient records."
    },
    {
        "question": "What are the risks of hallucination in medical AI?",
        "ground_truth": "Hallucination in medical AI refers to generation of fabricated information which poses risks including misdiagnosis and incorrect treatment recommendations."
    },
    {
        "question": "How does RAG reduce errors in medical question answering?",
        "ground_truth": "RAG reduces errors by retrieving relevant documents and grounding LLM responses in actual evidence rather than relying on parametric knowledge."
    },
    {
        "question": "What are challenges of deploying AI in clinical settings?",
        "ground_truth": "Challenges include data privacy, bias in training data, lack of interpretability, regulatory compliance, and integration with existing workflows."
    },
    {
        "question": "How is NLP used to extract information from electronic health records?",
        "ground_truth": "NLP extracts clinical entities such as diagnoses, medications and procedures from unstructured EHR text using named entity recognition models."
    },
]


def llm_judge(prompt):
    """Call Groq to act as an evaluator judge. Returns score 0.0-1.0."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    text = response.choices[0].message.content.strip()
    try:
        score = float(text)
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5


def score_faithfulness(question, answer, contexts):
    """
    Faithfulness: is every claim in the answer supported by the context?
    Score 1.0 = fully grounded, 0.0 = completely made up
    """
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an expert evaluator. Score whether the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context_text}

ANSWER:
{answer}

Rules:
- Score 1.0 if every claim in the answer is explicitly supported by the context
- Score 0.5 if the answer is partially supported
- Score 0.0 if the answer contains information not found in the context

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
    return llm_judge(prompt)


def score_answer_relevancy(question, answer):
    """
    Answer Relevancy: does the answer actually address the question?
    Score 1.0 = directly answers, 0.0 = completely off topic
    """
    prompt = f"""You are an expert evaluator. Score whether the ANSWER addresses the QUESTION.

QUESTION: {question}

ANSWER: {answer}

Rules:
- Score 1.0 if the answer directly and completely addresses the question
- Score 0.5 if the answer partially addresses the question
- Score 0.0 if the answer is irrelevant to the question

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
    return llm_judge(prompt)


def score_context_recall(question, contexts, ground_truth):
    """
    Context Recall: did the retriever find chunks containing the right info?
    Score 1.0 = retrieved context covers the ground truth answer
    """
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an expert evaluator. Score whether the CONTEXT contains enough information to answer the QUESTION correctly given the GROUND TRUTH.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
CONTEXT: {context_text}

Rules:
- Score 1.0 if the context contains all information needed to produce the ground truth answer
- Score 0.5 if the context is partially relevant
- Score 0.0 if the context is missing key information from the ground truth

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""
    return llm_judge(prompt)


def run_evaluation():
    print("Loading RAG system...")
    vectorstore      = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    print(f"\nEvaluating {len(eval_questions)} questions...\n")

    results = []

    for i, item in enumerate(eval_questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(eval_questions)}] {q[:60]}...")

        # Get RAG answer + retrieved chunks
        result       = ask(chain, retriever, q)
        answer       = result["answer"]
        raw_docs     = retriever.invoke(q)
        contexts     = [doc.page_content for doc in raw_docs]

        # Score each metric
        faith   = score_faithfulness(q, answer, contexts)
        relev   = score_answer_relevancy(q, answer)
        recall  = score_context_recall(q, contexts, gt)
        overall = (faith + relev + recall) / 3

        results.append({
            "question":         q,
            "faithfulness":     faith,
            "answer_relevancy": relev,
            "context_recall":   recall,
            "overall":          overall,
        })

        print(f"   Faithfulness: {faith:.2f} | Relevancy: {relev:.2f} | Recall: {recall:.2f} | Overall: {overall:.2f}")
        #print(f"   Faithfulness: {faith:.2f} | Relevancy: {relev:.2f} | Recall: {recall:.2f} | Overall: {overall:.2f}")
        time.sleep(15)  # wait 15 seconds between questions to avoid rate limit

    # ── Summary table ──────────────────────────────────────────
    avg_faith  = sum(r["faithfulness"]     for r in results) / len(results)
    avg_relev  = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_recall = sum(r["context_recall"]   for r in results) / len(results)
    avg_all    = (avg_faith + avg_relev + avg_recall) / 3

    print("\n" + "="*55)
    print("  EVALUATION RESULTS — PubMed Clinical RAG")
    print("="*55)
    print(f"  Faithfulness     : {avg_faith:.3f}  (answer grounded in docs?)")
    print(f"  Answer Relevancy : {avg_relev:.3f}  (answer addresses question?)")
    print(f"  Context Recall   : {avg_recall:.3f}  (retriever found right chunks?)")
    print("-"*55)
    print(f"  Overall Score    : {avg_all:.3f}")
    print("="*55)
    print("\nNote: Scores evaluated using LLM-as-a-judge (Groq LLaMA-3.1)")

    return results


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in .env file")
        exit(1)
    run_evaluation()