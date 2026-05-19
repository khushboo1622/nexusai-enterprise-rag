"""
evals/ragas_eval.py

Ragas evaluation pipeline for NexusAI.

What it does:
  1. Pulls query logs from MongoDB (stored silently during chat)
  2. Runs Ragas metrics: faithfulness, answer relevance, context recall
  3. Prints scores and saves results to evals/results.json

Metrics explained:
  - Faithfulness:       Is the answer grounded in retrieved context? (0-1)
  - Answer Relevance:   Does the answer address the question? (0-1)
  - Context Recall:     Were the right documents retrieved? (0-1)

Usage:
    python -m evals.ragas_eval
    python -m evals.ragas_eval --limit 20  (evaluate last 20 queries)
"""

import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

from backend.db.mongodb import get_chat_logs_collection
from backend.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


def fetch_logs(limit: int = 50) -> list[dict]:
    """
    Pull recent chat logs from MongoDB.
    Only includes non-blocked queries that have sources.
    """
    logs = get_chat_logs_collection()
    cursor = logs.find(
        {
            "blocked": {"$ne": True},          # skip guardrail-blocked queries
            "sources": {"$exists": True, "$ne": []},  # must have retrieved sources
            "answer": {"$exists": True, "$ne": ""},
        },
        sort=[("timestamp", -1)],
        limit=limit,
    )
    return list(cursor)


def prepare_ragas_dataset(logs: list[dict]) -> Dataset:
    """
    Convert MongoDB logs to Ragas Dataset format.

    Ragas expects:
      - question:   the user's question
      - answer:     the LLM's response
      - contexts:   list of retrieved text chunks (we stored sources, not full text)
      - ground_truth: expected answer (we don't have this — context_recall will be skipped)
    """
    questions = []
    answers = []
    contexts = []

    for log in logs:
        q = log.get("question", "").strip()
        a = log.get("answer", "").strip()
        srcs = log.get("sources", [])

        if not q or not a:
            continue

        # Build context list from stored source metadata
        # Note: we stored metadata not full text — use file_name as proxy
        # For proper context_recall you'd store full chunk text
        ctx = [
            f"{s.get('department', '')} / {s.get('file_name', '')} / {s.get('doc_type', '')}"
            for s in srcs
        ] or ["No context retrieved"]

        questions.append(q)
        answers.append(a)
        contexts.append(ctx)

    return Dataset.from_dict({
        "question":  questions,
        "answer":    answers,
        "contexts":  contexts,
    })


def run_eval(limit: int = 50):
    logger.info(f"Fetching last {limit} queries from MongoDB...")
    logs = fetch_logs(limit)

    if not logs:
        logger.error("No logs found. Run some queries first via the chat UI.")
        return

    logger.info(f"Found {len(logs)} queries to evaluate")
    dataset = prepare_ragas_dataset(logs)

    if len(dataset) == 0:
        logger.error("No valid queries to evaluate after filtering.")
        return

    logger.info(f"Running Ragas evaluation on {len(dataset)} queries...")

    # Run evaluation
    # Note: context_recall requires ground_truth — skipped here
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    # Print results
    print("\n" + "="*50)
    print("NEXUSAI RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Queries evaluated:    {len(dataset)}")
    print(f"Faithfulness:         {result['faithfulness']:.3f} (target: > 0.80)")
    print(f"Answer Relevancy:     {result['answer_relevancy']:.3f} (target: > 0.75)")
    print("="*50 + "\n")

    # Save results
    output = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "queries_evaluated": len(dataset),
        "scores": {
            "faithfulness":     result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
        },
        "targets": {
            "faithfulness":     0.80,
            "answer_relevancy": 0.75,
        },
        "passed": (
            result["faithfulness"] >= 0.80 and
            result["answer_relevancy"] >= 0.75
        ),
    }

    results_path = Path("evals/results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    if output["passed"]:
        logger.info("EVALUATION PASSED")
    else:
        logger.warning("EVALUATION FAILED — review answers for quality issues")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on NexusAI")
    parser.add_argument("--limit", type=int, default=50, help="Number of queries to evaluate")
    args = parser.parse_args()
    run_eval(limit=args.limit)