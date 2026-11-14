import json

chunks = json.load(open("data/chunks.json", encoding="utf-8"))
questions = json.load(open("data/questions.json", encoding="utf-8"))

answerable = [q for q in questions if q.get("answerable", True)]
print(f"Total chunks: {len(chunks)}")
print(f"Total answerable questions: {len(answerable)}")

mismatches = 0
for q in answerable:
    for g in q["gold"]:
        found = False
        for c in chunks:
            if c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]:
                found = True
                break
        if not found:
            mismatches += 1
            print(f"MISMATCH in {q['id']}: page {g['page']}, quote: {repr(g['evidence'][:40])}")

if mismatches == 0:
    print("ALL gold evidence quotes perfectly match within chunks!")
else:
    print(f"Found {mismatches} mismatches.")
