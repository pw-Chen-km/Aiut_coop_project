import numpy as np
import pytest

from topic_rag.retrieval import Retriever, rank_topics, topic_mass
from topic_rag.taxonomy import Taxonomy


def retriever():
    chunks = [{"chunk_id": f"c{i:02d}", "source_id": f"s{i}", "text": "x",
               "retrieval_text": "wifi" if i < 5 else "remote"} for i in range(8)]
    labels = {f"s{i}": {"topic_options": ["2"] if i == 7 else ["1"]} for i in range(8)}
    vectors = np.array([[1, i * .1] for i in range(8)], dtype=np.float32)
    return Retriever(chunks, vectors, labels, Taxonomy({"1": "network", "2": "remote"}))


def test_hard_filters_before_candidates_with_global_idf():
    r = retriever()
    before = dict(r.bm25.idf)
    baseline, _ = r.retrieve("wifi", [1, 0], "baseline", candidate_k=2)
    hard, trace = r.retrieve("wifi", [1, 0], "hard_topic", {"1": .1, "2": .9}, candidate_k=2)
    assert "c07" not in {c["chunk_id"] for c in baseline}
    assert [c["chunk_id"] for c in hard] == ["c07"]
    assert trace["allowed_chunk_count"] == 1
    assert before == r.bm25.idf


def test_soft_keeps_same_pool_and_exact_equal_weights():
    r = retriever()
    _, base_trace = r.retrieve("wifi", [0, 1], "baseline", candidate_k=2)
    soft, trace = r.retrieve("wifi", [0, 1], "soft_topic", {"1": .3, "2": .7}, candidate_k=2)
    assert trace["candidate_ids"] == base_trace["candidate_ids"]
    assert trace["candidate_count"] <= 4
    missing_rank_seen = False
    for item in soft:
        for name, rank in item["component_ranks"].items():
            expected = (1 / 3) / (60 + rank) if rank is not None else 0
            assert item["component_scores"][name] == pytest.approx(expected)
            missing_rank_seen |= rank is None
        assert item["score"] == pytest.approx(sum(item["component_scores"].values()))
    assert missing_rank_seen
    assert soft == r.retrieve("wifi", [0, 1], "soft_topic", {"1": .3, "2": .7}, candidate_k=2)[0]


def test_multilabel_mass_is_sum_and_ties_share_rank():
    assert topic_mass(["1", "2", "2"], {"1": .3, "2": .7}) == pytest.approx(1)
    assert rank_topics({"a": .8, "b": .8, "c": .1}) == {"a": 1, "b": 1, "c": 3}


@pytest.mark.parametrize("probabilities", [{"T01": .5, "T02": .5}, {"1": .2, "2": .2},
                                          {"1": float("nan"), "2": 1}, None])
def test_bad_distributions_rejected(probabilities):
    with pytest.raises(ValueError):
        retriever().retrieve("wifi", [1, 0], "soft_topic", probabilities)
