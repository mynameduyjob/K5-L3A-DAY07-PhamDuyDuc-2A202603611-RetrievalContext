"""Prepare chunks and benchmark questions for Day 7 from Day 6 assets."""
import json
import hashlib
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    paragraphs = []
    for line in lines:
        if line:
            paragraphs.append(line)
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    return "\n".join(paragraphs).strip()

def load_text_units(path: Path, document_id: str, source: str, units: list, metadata: dict) -> list:
    text = normalize_text(path.read_text(encoding="utf-8-sig"))
    pages = []
    for unit in units:
        start, end = unit["start_char"], unit["end_char"]
        value = text[start:end]
        unit_meta = dict(metadata or {})
        unit_meta.update({
            "unit_id": unit["unit_id"],
            "chapter": unit["chapter"],
            "char_start": start,
            "char_end": end,
            "char_count": len(value),
            "location_type": "text_unit",
            "extraction_status": "ok",
        })
        pages.append({
            "document_id": document_id,
            "source": source,
            "page": unit["page"],
            "text": value,
            "metadata": unit_meta,
        })
    return pages

def by_paragraph(page: dict, max_size: int = 1000) -> list:
    text = page["text"].strip()
    chunks = []
    group_start = None
    group_end = None

    def make_chunk(start, end, chunk_text):
        identity = f"{page['document_id']}:{page['page']}:{start}:{end}:{chunk_text}"
        cid = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        meta = dict(page["metadata"])
        meta.update({"char_start": start, "char_end": end})
        return {
            "id": cid,
            "document_id": page["document_id"],
            "source_id": page["document_id"],
            "source": page["source"],
            "page": page["page"],
            "text": chunk_text,
            "metadata": meta,
        }

    def flush():
        nonlocal group_start, group_end
        if group_start is not None:
            chunks.append(make_chunk(group_start, group_end, text[group_start:group_end]))
        group_start, group_end = None, None

    cursor = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            cursor += 2
            continue
        para_start = text.find(paragraph, cursor)
        if para_start < 0:
            para_start = cursor
        para_end = para_start + len(paragraph)
        cursor = para_end + 2

        if len(paragraph) > max_size:
            flush()
            for start in range(para_start, para_end, max_size):
                end = min(start + max_size, para_end)
                chunks.append(make_chunk(start, end, text[start:end]))
            continue

        if group_start is not None and para_end - group_start > max_size:
            flush()
        if group_start is None:
            group_start = para_start
        group_end = para_end
    flush()
    return chunks

def prepare_chunks():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    txt_path = DATA_DIR / "huck_finn_pg76.txt"
    pages = load_text_units(
        txt_path,
        source["source_id"],
        source["title"],
        source["included_units"],
        source["metadata"],
    )
    chunks = []
    for p in pages:
        chunks.extend(by_paragraph(p, max_size=1000))
    print(f"Generated {len(chunks)} chunks from {len(pages)} units.")
    out_path = DATA_DIR / "chunks.json"
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks

def prepare_questions(chunks):
    import re
    day6_questions = json.loads((DATA_DIR / "benchmark_day6.json").read_text(encoding="utf-8"))
    questions = []
    for q in day6_questions:
        gold = q.get("gold")
        if not gold:
            raw_evidence = q.get("evidence_quote")
            gold_page = q.get("gold_page")
            gold_source_id = q.get("gold_source_id")
            # Find the exact text slice in chunks to satisfy baseline.py exact substring requirement
            words = raw_evidence.split()
            pattern = r"\s+".join(re.escape(w) for w in words)
            exact_evidence = raw_evidence
            for c in chunks:
                if c["source_id"] == gold_source_id and c["page"] == gold_page:
                    m = re.search(pattern, c["text"], flags=re.IGNORECASE)
                    if m:
                        exact_evidence = c["text"][m.start():m.end()]
                        break
            gold = [{
                "source_id": gold_source_id,
                "page": gold_page,
                "evidence": exact_evidence,
            }]
        record = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold": gold,
            "expected_answer": q["expected_answer"],
            "answerable": True,
        }
        if "filter" in q and q["filter"]:
            record["filter"] = q["filter"]
        questions.append(record)

    # Add 4 unanswerable questions (2 dev, 2 test)
    unanswerable = [
        {
            "id": "q_unans_dev_01",
            "split": "dev",
            "question": "In Chapter IV, what color was Jim's ox-hair ball and where did he say he found it?",
            "expected_answer": "I do not have sufficient evidence in the text to answer this question.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Chapter IV mentions Jim has an ox hair-ball, but never specifies its color or where he found it.",
            "gold": [],
        },
        {
            "id": "q_unans_dev_02",
            "split": "dev",
            "question": "How much money did Judge Thatcher offer to pay Pap in Chapter V to buy full custody of Huck?",
            "expected_answer": "I do not have sufficient evidence in the text to answer this question.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "In Chapter V, Judge Thatcher and the widow went to court for custody, but Thatcher never offered or paid money to Pap.",
            "gold": [],
        },
        {
            "id": "q_unans_test_01",
            "split": "test",
            "question": "In Chapter X, what was the full name of the doctor who treated Jim's rattlesnake bite on Jackson's Island?",
            "expected_answer": "I do not have sufficient evidence in the text to answer this question.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "In Chapter X, no doctor was ever called or visited Jackson's Island; Huck and Jim treated the snake bite themselves.",
            "gold": [],
        },
        {
            "id": "q_unans_test_02",
            "split": "test",
            "question": "In Chapter XI, how many children did Mrs. Judith Loftus mention living in her cabin?",
            "expected_answer": "I do not have sufficient evidence in the text to answer this question.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "In Chapter XI, Huck talks with Mrs. Judith Loftus, but she never mentions how many children she has.",
            "gold": [],
        },
    ]
    questions.extend(unanswerable)
    print(f"Total benchmark questions: {len(questions)} (Dev: {sum(1 for q in questions if q['split'] == 'dev')}, Test: {sum(1 for q in questions if q['split'] == 'test')}).")
    out_path = DATA_DIR / "questions.json"
    out_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also save answerable-only questions for starter/baseline.py validate()
    answerable_only = [q for q in questions if q.get("answerable", True)]
    (DATA_DIR / "questions_answerable.json").write_text(json.dumps(answerable_only, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also save cases.json for starter/agent_metrics.py
    cases = [{"id": q["id"], "split": q["split"], "answerable": q.get("answerable", True)} for q in questions]
    (DATA_DIR / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return questions

if __name__ == "__main__":
    chunks = prepare_chunks()
    prepare_questions(chunks)
