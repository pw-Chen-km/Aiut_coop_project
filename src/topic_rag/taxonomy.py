"""Strict numeric topic IDs and annotation validation."""
import json
import math
from pathlib import Path

from topic_rag.io import digest_text


class Taxonomy:
    def __init__(self, options):
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError("at least two numeric options required")
        expected = [str(i) for i in range(1, len(options) + 1)]
        if set(options) != set(expected) or any(not isinstance(v, str) or not v.strip() for v in options.values()):
            raise ValueError("option IDs must be consecutive strings: 1, 2, ...")
        self.options = {key: options[key] for key in expected}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["options"])

    def check_distribution(self, probabilities):
        if not isinstance(probabilities, dict) or set(probabilities) != set(self.options):
            raise ValueError("probability keys differ from taxonomy")
        if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in probabilities.values()):
            raise ValueError("probabilities must be finite and nonnegative")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")

    def check_topics(self, topics):
        if not isinstance(topics, list) or not topics or len(set(topics)) != len(topics):
            raise ValueError("topics must be a nonempty unique list")
        if any(key not in self.options for key in topics):
            raise ValueError("unknown topic ID")


def validate_labels(sources, labels, taxonomy):
    by_id = {}
    for label in labels:
        sid = label["source_id"]
        if sid in by_id:
            raise ValueError(f"duplicate label: {sid}")
        taxonomy.check_topics(label["topic_options"])
        by_id[sid] = label
    if set(by_id) != {row["source_id"] for row in sources}:
        raise ValueError("source/annotation IDs differ")
    for source in sources:
        if by_id[source["source_id"]]["source_sha256"] != digest_text(source["text"]):
            raise ValueError("annotation is stale: source text changed")
        excerpt = by_id[source["source_id"]].get("supporting_excerpt")
        if excerpt is not None and (not excerpt or excerpt not in source["text"]):
            raise ValueError("annotation excerpt is not in the source")
    return by_id

