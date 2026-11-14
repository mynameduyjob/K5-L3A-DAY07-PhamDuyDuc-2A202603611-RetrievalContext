"""Benchmark the full Agentic RAG pipeline on dev and test splits."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import AgentController, run_agent_loop
from retrievers import (
    ChromaDenseRetriever,
    HybridRetriever,
    TfidfRetriever,
    hit_check,
)
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def run_agent_eval():
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))

    print("Initializing retrievers for agent loop...")
    tfidf = TfidfRetriever(chunks)
    dense = ChromaDenseRetriever(chunks)
    hybrid = HybridRetriever(dense, tfidf)

    api_key = os.getenv("GEMINI_API_KEY", "")
    controller = AgentController(api_key=api_key)

    for split in ("dev", "test"):
        split_questions = [q for q in questions if q["split"] == split]
        print(f"\nEvaluating Agent on {split} ({len(split_questions)} questions)...")

        traces = []
        reviews = []
        agent_out_dir = RESULTS_DIR / f"agent_{split}"
        agent_out_dir.mkdir(parents=True, exist_ok=True)

        for q in split_questions:
            qid = q["id"]
            question_text = q["question"]
            filters = q.get("filter")
            gold = q.get("gold", [])
            answerable = q.get("answerable", True)

            t0 = time.perf_counter()
            res = run_agent_loop(
                question=question_text,
                controller=controller,
                retriever=hybrid,
                max_searches=3,
                top_k=3,
                budget=1800,
                filters=filters,
            )
            elapsed_sec = time.perf_counter() - t0

            trace_record = {
                "query_id": qid,
                "split": split,
                "question": question_text,
                "answerable": answerable,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "events": res["events"],
                "elapsed_seconds": elapsed_sec,
            }
            traces.append(trace_record)

            # Build review record
            citations = res["citations"]
            # Check citation support: did cited chunk contain gold evidence?
            citation_support = []
            for cid in citations:
                chunk = hybrid.chunk_lookup.get(cid)
                is_supported = bool(chunk and hit_check(chunk, gold))
                citation_support.append(is_supported)

            if answerable:
                # Grade answer:
                # 2 = answered with supported citation
                # 1 = answered but citation missing or partially supported
                # 0 = abstained or wrong citation
                if res["status"] == "abstain":
                    answer_grade = 0
                elif all(citation_support) and citation_support:
                    answer_grade = 2
                elif any(citation_support):
                    answer_grade = 1
                else:
                    answer_grade = 0

                reviews.append({
                    "query_id": qid,
                    "answer_grade": answer_grade,
                    "citation_support": citation_support,
                    "notes": f"Answerable question; status={res['status']}, grade={answer_grade}",
                })
            else:
                # Unanswerable question
                abstention_correct = (res["status"] == "abstain")
                reviews.append({
                    "query_id": qid,
                    "abstention_correct": abstention_correct,
                    "citation_support": [],
                    "notes": f"Unanswerable question; status={res['status']}, abstention_correct={abstention_correct}",
                })

        (agent_out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (agent_out_dir / "reviews.json").write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Call starter/agent_metrics.py aggregate function
        from starter.agent_metrics import aggregate
        cases = [c for c in json.loads((DATA_DIR / "cases.json").read_text(encoding="utf-8")) if c["split"] == split]
        report = {"split": split, **aggregate(cases, traces, reviews)}
        (agent_out_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Summary for {split}:")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_agent_eval()
