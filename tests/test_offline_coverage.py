from offline.ingest.build import coverage


def test_coverage_rejects_source_gaps():
    corpus = {
        "blocks": [{"block_id": "b1", "text": "abcdef", "build_role": "content"}],
        "chunks": [{"source_spans": [{"block_id": "b1", "source_start_char": 0, "source_end_char": 3}]}],
    }
    report = coverage(corpus)
    assert not report["valid"]
    assert report["failures"][0]["from"] == 3


def test_coverage_accepts_exact_partition():
    corpus = {
        "blocks": [{"block_id": "b1", "text": "abcdef", "build_role": "content"}],
        "chunks": [{"source_spans": [
            {"block_id": "b1", "source_start_char": 0, "source_end_char": 3},
            {"block_id": "b1", "source_start_char": 3, "source_end_char": 6},
        ]}],
    }
    assert coverage(corpus)["valid"]
