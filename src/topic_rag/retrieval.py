"""Shared BM25/dense candidates and exact equal-weight topic RRF."""
from collections import Counter
import math
import re

import numpy as np


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.casefold())


class BM25:
    def __init__(self, chunks, k1=1.2, b=0.75):
        self.tf = [Counter(tokenize(c["retrieval_text"])) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        df = Counter(t for tf in self.tf for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.avg = sum(self.lengths) / n if n else 1
        self.k1, self.b = k1, b

    def scores(self, query, indices=None):
        terms = Counter(tokenize(query))
        selected = range(len(self.tf)) if indices is None else indices
        return [sum(
            count * self.idf.get(term, 0) * self.tf[i].get(term, 0) * (self.k1 + 1)
            / (self.tf[i].get(term, 0) + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1)))
            for term, count in terms.items()
        ) for i in selected]


def topic_mass(topics, probabilities):
    return math.fsum(probabilities[t] for t in set(topics))


def rank_topics(scores):
    # Equal topic mass must not create arbitrary rank differences. Competition
    # ranks (1,1,3,...) give tied chunks the same topic contribution.
    return {cid: 1 + sum(other > score for other in scores.values()) for cid, score in scores.items()}


class Retriever:
    def __init__(self, chunks, vectors, labels, taxonomy):
        self.chunks, self.labels, self.taxonomy = chunks, labels, taxonomy
        self.by_id = {c["chunk_id"]: c for c in chunks}
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("invalid dense vector matrix")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0):
            raise ValueError("zero dense vector")
        self.vectors = vectors / norms[:, None]
        self.bm25 = BM25(chunks)

    def retrieve(self, query, query_vector, variant, probabilities=None, candidate_k=20, output_k=20):
        if candidate_k < 1 or output_k < 1 or variant not in {"baseline", "hard_topic", "soft_topic"}:
            raise ValueError("invalid retrieval configuration")
        selected = None
        if variant != "baseline":
            self.taxonomy.check_distribution(probabilities)
            selected = max(self.taxonomy.options, key=lambda t: probabilities[t])
        allowed = [i for i, c in enumerate(self.chunks) if variant != "hard_topic"
                   or selected in self.labels[c["source_id"]]["topic_options"]]
        q = np.asarray(query_vector, dtype=np.float32)
        if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError("invalid query vector")
        dense_scores = dict(zip(allowed, self.vectors[allowed] @ (q / np.linalg.norm(q))))
        lexical_scores = dict(zip(allowed, self.bm25.scores(query, allowed)))
        # Keep corpus-wide IDF identical across methods; mask before top-k.
        def ranked(scores):
            order = sorted(allowed, key=lambda i: (-float(scores[i]), self.chunks[i]["chunk_id"]))[:candidate_k]
            return {self.chunks[i]["chunk_id"]: rank for rank, i in enumerate(order, 1)}
        lexical, dense = ranked(lexical_scores), ranked(dense_scores)
        union = set(lexical) | set(dense)
        masses = {cid: topic_mass(self.labels[self.by_id[cid]["source_id"]]["topic_options"], probabilities)
                  for cid in union} if variant == "soft_topic" else {}
        topics = rank_topics(masses) if masses else {}
        results = []
        for cid in sorted(union):
            ranks = {"bm25": lexical.get(cid), "e5": dense.get(cid)}
            weight = 0.5
            if variant == "soft_topic":
                weight = 1 / 3
                ranks["topic"] = topics[cid]
            contributions = {name: weight / (60 + rank) if rank is not None else 0.0
                             for name, rank in ranks.items()}
            results.append({**self.by_id[cid], "score": sum(contributions.values()),
                            "component_ranks": ranks, "component_scores": contributions,
                            "topic_mass": masses.get(cid)})
        results.sort(key=lambda r: (-r["score"], r["chunk_id"]))
        for rank, row in enumerate(results, 1):
            row["rank"] = rank
        return results[:output_k], {
            "selected_topic": selected, "allowed_chunk_count": len(allowed),
            "candidate_ids": sorted(union), "candidate_count": len(union),
        }

