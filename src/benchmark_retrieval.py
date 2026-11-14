"""Benchmark retrieval methods on dev and test splits for Day 7."""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrievers import (
    ChromaDenseRetriever,
    HybridRetriever,
    TfidfRetriever,
    hit_check,
    pack_context,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_retriever(
    retriever_name: str,
    search_fn,
    questions: List[Dict[str, Any]],
    budget: int = 800,
    use_filter: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    traces = []
    times_ms = []

    for q in questions:
        filters = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        results = search_fn(q["question"], top_k=3, filters=filters)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

        context, ids = pack_context(results, budget=budget)
        gold = q.get("gold", [])

        marked = [{**c, "hit": hit_check(c, gold)} for c in results]
        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(c["hit"] for c in marked)
        context_hit = any(c["hit"] and c["id"] in ids for c in marked)

        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": retriever_name,
            "top_k": 3,
            "budget_chars": budget,
            "filters": filters,
            "gold": gold,
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
        })

    n = len(questions)
    hit1_count = sum(t["hit1"] for t in traces)
    hit3_count = sum(t["hit3"] for t in traces)
    context_hit_count = sum(t["context_hit"] for t in traces)

    summary = {
        "strategy": retriever_name,
        "split": questions[0]["split"] if questions else "dev",
        "n": n,
        "budget_chars": budget,
        "use_filter": use_filter,
        "hit1": hit1_count,
        "hit1_rate": hit1_count / n if n else 0.0,
        "hit3": hit3_count,
        "hit3_rate": hit3_count / n if n else 0.0,
        "context_evidence_count": context_hit_count,
        "context_evidence_rate": context_hit_count / n if n else 0.0,
        "avg_context_chars": statistics.mean(t["context_chars"] for t in traces) if traces else 0.0,
        "median_ms": statistics.median(times_ms) if times_ms else 0.0,
        "mean_ms": statistics.mean(times_ms) if times_ms else 0.0,
        "cost_usd": 0.0,
    }
    return summary, traces


def run_all_benchmarks():
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    all_questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))

    # Answerable questions for retrieval evaluation
    dev_answerable = [q for q in all_questions if q["split"] == "dev" and q.get("answerable", True)]
    test_answerable = [q for q in all_questions if q["split"] == "test" and q.get("answerable", True)]

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Dev answerable questions: {len(dev_answerable)}")
    print(f"Test answerable questions: {len(test_answerable)}")

    print("\n--- Initializing Retrievers ---")
    t0 = time.perf_counter()
    tfidf = TfidfRetriever(chunks)
    tfidf_init_sec = time.perf_counter() - t0
    print(f"TfidfRetriever initialized in {tfidf_init_sec:.3f}s")

    t0 = time.perf_counter()
    dense = ChromaDenseRetriever(chunks)
    dense_init_sec = time.perf_counter() - t0
    print(f"ChromaDenseRetriever initialized in {dense_init_sec:.3f}s")

    hybrid = HybridRetriever(dense, tfidf)

    experiments = [
        # DEV EXPERIMENTS
        {
            "name": "baseline_tfidf",
            "search_fn": tfidf.search,
            "questions": dev_answerable,
            "budget": 800,
            "use_filter": False,
            "desc": "Baseline Lexical TF-IDF (budget 800)"
        },
        {
            "name": "dense_chroma",
            "search_fn": dense.search,
            "questions": dev_answerable,
            "budget": 800,
            "use_filter": False,
            "desc": "Method 2: Dense ChromaDB all-MiniLM-L6-v2 (budget 800)"
        },
        {
            "name": "hybrid_rrf",
            "search_fn": hybrid.search,
            "questions": dev_answerable,
            "budget": 800,
            "use_filter": False,
            "desc": "Method 3: Hybrid Dense+TF-IDF via RRF (budget 800)"
        },
        {
            "name": "improvement1_budget1800",
            "search_fn": hybrid.search,
            "questions": dev_answerable,
            "budget": 1800,
            "use_filter": False,
            "desc": "Improvement 1: Hybrid RRF with expanded budget (1800 chars)"
        },
        {
            "name": "improvement2_filter_budget1800",
            "search_fn": hybrid.search,
            "questions": dev_answerable,
            "budget": 1800,
            "use_filter": True,
            "desc": "Improvement 2: Hybrid RRF + Metadata Chapter Filter + budget 1800"
        },
        # TEST EVALUATION
        {
            "name": "test_baseline_tfidf",
            "search_fn": tfidf.search,
            "questions": test_answerable,
            "budget": 800,
            "use_filter": False,
            "desc": "Test Baseline: TF-IDF (budget 800)"
        },
        {
            "name": "test_chosen_best",
            "search_fn": hybrid.search,
            "questions": test_answerable,
            "budget": 1800,
            "use_filter": True,
            "desc": "Test Chosen: Hybrid RRF + Filter + budget 1800"
        },
    ]

    all_summaries = {}
    for exp in experiments:
        name = exp["name"]
        print(f"\nRunning experiment: {exp['desc']}...")
        summary, traces = evaluate_retriever(
            retriever_name=name,
            search_fn=exp["search_fn"],
            questions=exp["questions"],
            budget=exp["budget"],
            use_filter=exp["use_filter"],
        )
        all_summaries[name] = summary
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" -> Hit@1: {summary['hit1']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%), "
              f"Hit@3: {summary['hit3']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%), "
              f"Context Hit: {summary['context_evidence_count']}/{summary['n']} ({summary['context_evidence_rate']*100:.1f}%), "
              f"Median Latency: {summary['median_ms']:.2f}ms")

    (RESULTS_DIR / "all_retrieval_summaries.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAll retrieval experiments completed successfully!")


if __name__ == "__main__":
    from typing import Tuple
    run_all_benchmarks()
