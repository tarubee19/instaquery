"""Retrieval of similar (question, SQL) training examples for retrieval-augmented few-shot prompting.

Default: dense embeddings (BAAI/bge-small-en-v1.5 via sentence-transformers).
Fallback: TF-IDF cosine similarity (no model download needed).
"""

from __future__ import annotations

import numpy as np


class ExampleRetriever:
    def __init__(self, pool: list[dict], model_name: str | None = "BAAI/bge-small-en-v1.5",
                 device: str | None = None):
        self.pool = pool
        self.questions = [p["question"] for p in pool]
        self.method = "tfidf"
        self._st = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name, device=device)
                self._emb = self._encode(self.questions)
                self.method = f"dense:{model_name}"
            except Exception as e:  # pragma: no cover - depends on network/model
                print(f"[retriever] dense model unavailable ({e}); using TF-IDF")
                self._st = None
        if self._st is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self._mat = self._tfidf.fit_transform(self.questions)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._st.encode(texts, batch_size=128, normalize_embeddings=True,
                                          show_progress_bar=False), dtype=np.float32)

    def retrieve_many(self, questions: list[str], k: int = 3, exclude_db: list[str] | None = None) -> list[list[dict]]:
        if self._st is not None:
            q = self._encode(questions)
            sims = q @ self._emb.T
        else:
            sims = (self._tfidf.transform(questions) @ self._mat.T).toarray()
        results = []
        for i in range(len(questions)):
            order = np.argsort(-sims[i])
            picked, seen_sql = [], set()
            for j in order:
                cand = self.pool[int(j)]
                if exclude_db and cand["db_id"] == exclude_db[i]:
                    continue  # never leak the same database
                if cand["query"] in seen_sql:
                    continue
                picked.append(cand)
                seen_sql.add(cand["query"])
                if len(picked) == k:
                    break
            results.append(picked)
        return results


def pick_fixed_shots(train: list[dict], k: int = 3) -> list[dict]:
    """Deterministically pick k diverse examples from the *training* set:
    a filter+aggregate, a GROUP BY/HAVING, and a JOIN with ORDER BY/LIMIT."""
    import re
    patterns = [r"^SELECT count\(\*\) FROM \w+ WHERE",
                r"GROUP BY .* HAVING",
                r"JOIN .* ORDER BY .* LIMIT 1$"]
    shots = []
    for pat in patterns:
        for ex in train:
            q = " ".join(ex["query"].split())
            if re.search(pat, q, re.I) and len(q) < 160 and ex not in shots:
                shots.append(ex)
                break
    for ex in train:  # top up if a pattern was not found
        if len(shots) >= k:
            break
        if ex not in shots:
            shots.append(ex)
    return shots[:k]
