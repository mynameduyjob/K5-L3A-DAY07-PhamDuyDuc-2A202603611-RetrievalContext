"""Retrieval implementations for Day 7: Dense, Lexical (TF-IDF), and Hybrid (RRF)."""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())

def hit_check(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    return any(
        chunk.get("source_id") == g.get("source_id")
        and chunk.get("page") == g.get("page")
        and g.get("evidence", "") in chunk.get("text", "")
        for g in gold
    )

def pack_context(results: List[Dict[str, Any]], budget: int = 800) -> Tuple[str, List[str]]:
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source', '')} | page {c.get('page', '')}]\n{c.get('text', '')}"
        candidate = "\n\n".join(blocks + [block]) if blocks else block
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(cid)
    return "\n\n".join(blocks), ids


class TfidfRetriever:
    """Lexical retrieval baseline using TF-IDF cosine similarity."""

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> Dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def search(self, question: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = self._vector(Counter(tokenize(question)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                meta = chunk.get("metadata", {})
                if not all(str(meta.get(k)) == str(v) for k, v in filters.items()):
                    continue
            score = sum(value * vector.get(term, 0.0) for term, value in query.items())
            if score > 0:
                results.append({**chunk, "score": score, "strategy": "tfidf"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class ChromaDenseRetriever:
    """Dense retrieval using ChromaDB with local ONNX all-MiniLM-L6-v2."""

    def __init__(self, chunks: List[Dict[str, Any]], persist_dir: Optional[Path] = None) -> None:
        import chromadb
        self.chunks = chunks
        self.chunk_lookup = {c["id"]: c for c in chunks}
        
        if persist_dir:
            self.client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self.client = chromadb.Client()
            
        collection_name = "huck_finn_chunks"
        # Reset collection if exists to guarantee freshness
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
            
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Batch insert chunks
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c["id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = []
            for c in batch:
                meta = dict(c.get("metadata", {}))
                meta["source_id"] = str(c.get("source_id", ""))
                meta["source"] = str(c.get("source", ""))
                meta["page"] = int(c.get("page", 1))
                metadatas.append(meta)
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, question: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        where_filter = None
        if filters:
            if len(filters) == 1:
                k, v = next(iter(filters.items()))
                where_filter = {k: {"$eq": str(v)}}
            else:
                where_filter = {"$and": [{k: {"$eq": str(v)}} for k, v in filters.items()]}

        res = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, len(self.chunks)),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        if res and res["ids"] and res["ids"][0]:
            ids = res["ids"][0]
            distances = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(ids)
            for cid, dist in zip(ids, distances):
                c = self.chunk_lookup[cid]
                # Cosine distance to similarity score: score = 1 - distance
                score = 1.0 - float(dist)
                results.append({**c, "score": score, "strategy": "chroma_dense"})
        return results


class HybridRetriever:
    """Hybrid retrieval combining Dense and TF-IDF via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: ChromaDenseRetriever, tfidf_retriever: TfidfRetriever, rrf_k: int = 60) -> None:
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever
        self.rrf_k = rrf_k
        self.chunk_lookup = self.dense.chunk_lookup

    def search(self, question: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None, candidate_k: int = 10) -> List[Dict[str, Any]]:
        dense_hits = self.dense.search(question, top_k=candidate_k, filters=filters)
        tfidf_hits = self.tfidf.search(question, top_k=candidate_k, filters=filters)

        rrf_scores: Dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, hit in enumerate(tfidf_hits, start=1):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])[:top_k]
        results = []
        for cid in sorted_ids:
            chunk = self.chunk_lookup[cid]
            results.append({**chunk, "score": rrf_scores[cid], "strategy": "hybrid_rrf"})
        return results
