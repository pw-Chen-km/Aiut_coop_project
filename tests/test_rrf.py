from qursor_enterprise_qa.rrf_retrieval import (
    BM25Index,
    SearchResult,
    weighted_rrf,
    weighted_rrf_many,
)


def test_bm25_and_rrf_are_deterministic():
    documents = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "restart qursor server procedure"},
        {"chunk_id": "c2", "doc_id": "d2", "text": "mqtt integration topic"},
        {"chunk_id": "c3", "doc_id": "d3", "text": "ssl installation procedure"},
    ]
    lexical = BM25Index(documents).search("restart server", 3)
    assert lexical[0].chunk_id == "c1"
    dense = [
        SearchResult("c2", "d2", documents[1]["text"], .9, 1, documents[1]),
        SearchResult("c1", "d1", documents[0]["text"], .8, 2, documents[0]),
        SearchResult("c3", "d3", documents[2]["text"], .7, 3, documents[2]),
    ]
    fused = weighted_rrf(lexical, dense, k=2, rrf_k=60)
    assert [item.chunk_id for item in fused] == ["c1", "c2"]
    assert fused[0].component_scores.keys() == {"bm25", "e5"}


def test_bm25_indexes_retrieval_metadata_without_changing_source_text():
    documents = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "copy the certificate files",
            "retrieval_text": (
                "document title: SSL Installation Manual\n"
                "filename: SSL-Installation-Manual.pdf\n"
                "section path: SSL Installation Manual > Installation steps\n"
                "copy the certificate files"
            ),
        },
        {"chunk_id": "c2", "doc_id": "d2", "text": "restart the server"},
    ]

    results = BM25Index(documents).search("Traefik SSL installation", 2)

    assert results[0].chunk_id == "c1"
    assert results[0].text == documents[0]["text"]


def test_rrf_returns_exact_final_k_when_candidates_exist():
    docs = [{"chunk_id": f"c{i}", "doc_id": "d", "text": f"term {i}"} for i in range(30)]
    lexical = BM25Index(docs).search("term", 20)
    dense = list(reversed(lexical))
    assert len(weighted_rrf(lexical, dense, k=5)) == 5


def test_weighted_rrf_many_uses_four_query_paths_and_stable_order():
    original_bm25 = [SearchResult("a", "d", "a", 1.0, 1), SearchResult("b", "d", "b", 1.0, 2)]
    original_e5 = [SearchResult("b", "d", "b", 1.0, 1), SearchResult("c", "d", "c", 1.0, 2)]
    rewrite_bm25 = [SearchResult("c", "d", "c", 1.0, 1), SearchResult("d", "d", "d", 1.0, 2)]
    rewrite_e5 = [SearchResult("d", "d", "d", 1.0, 1), SearchResult("a", "d", "a", 1.0, 2)]
    result = weighted_rrf_many(
        [
            ("original_bm25", original_bm25, 0.30),
            ("original_e5", original_e5, 0.30),
            ("rewrite_bm25", rewrite_bm25, 0.20),
            ("rewrite_e5", rewrite_e5, 0.20),
        ],
        k=4,
        rrf_k=60,
    )
    assert [item.chunk_id for item in result] == ["b", "a", "c", "d"]
    assert set(result[0].component_scores) == {"original_bm25", "original_e5"}
