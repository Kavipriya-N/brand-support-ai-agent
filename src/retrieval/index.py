"""
Retrieval Index Module for Grounded Reply Evidence.

Builds and queries an in-memory vector index over historical (Customer Inquiry -> Brand Resolution) pairs.
Supports fast top-k cosine similarity search and retrieval support score estimation.
"""

import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResolutionRetrievalIndex:
    def __init__(self, config_path: str = "configs/default.yaml", index_path: str = "data/processed/retrieval_index.pkl"):
        self.config_path = config_path
        self.index_path = Path(index_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.top_k = self.cfg["retrieval"].get("top_k", 3)
        self.min_relevance = self.cfg["retrieval"].get("min_relevance_threshold", 0.18)

        self.vectorizer = None
        self.matrix = None
        self.corpus_df = None

    def build_or_load(self, corpus_path: str = "data/processed/retrieval_corpus.parquet"):
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.matrix = data["matrix"]
                self.corpus_df = data["corpus_df"]
            return self

        print(f"[*] Building Retrieval Index over {corpus_path}...")
        self.corpus_df = pd.read_parquet(corpus_path).reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=12000,
            stop_words="english",
            sublinear_tf=True
        )

        texts = self.corpus_df["customer_text_clean"].tolist()
        self.matrix = self.vectorizer.fit_transform(texts)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "corpus_df": self.corpus_df
            }, f)

        print(f"[OK] Retrieval Index built and saved ({self.matrix.shape[0]} documents, {self.matrix.shape[1]} features) to {self.index_path}")
        return self

    def query(self, text: str, top_k: int = None) -> List[Dict[str, Any]]:
        if self.matrix is None:
            self.build_or_load()

        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            row = self.corpus_df.iloc[idx]
            results.append({
                "customer_tweet_id": str(row["customer_tweet_id"]),
                "brand_tweet_id": str(row["brand_tweet_id"]),
                "customer_text": str(row["customer_text_clean"]),
                "brand_reply": str(row["brand_reply_clean"]),
                "similarity": round(score, 4),
                "is_dm_escalation": bool(row.get("is_dm_escalation", False)),
                "has_followup": bool(row.get("has_followup", False))
            })

        return results

    def get_support_score(self, text: str, top_k: int = 3) -> float:
        results = self.query(text, top_k=top_k)
        if not results:
            return 0.0
        # Return top-1 similarity as primary support metric
        return results[0]["similarity"]

if __name__ == "__main__":
    idx = ResolutionRetrievalIndex()
    idx.build_or_load()
    q = "Package was supposed to arrive today but tracking shows delayed in transit"
    hits = idx.query(q, top_k=2)
    print(f"Query: '{q}'\nTop Hits:")
    for h in hits:
        print(f"  [Sim: {h['similarity']}] Issue: {h['customer_text'][:60]}... -> Reply: {h['brand_reply'][:60]}...")
