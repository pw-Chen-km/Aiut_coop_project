import json
import math

import httpx
import pytest

from topic_rag.model_client import ModelClient
from topic_rag.taxonomy import Taxonomy
from topic_rag.topic_scoring import JevScorer, ScoringUnavailable, extract_logprobs, score_trie, suffix_paths


def test_full_suffix_scoring_includes_closing_bracket_and_multidigit_option():
    # Shared prefix for 1] and 10], and conditional bracket likelihood matters.
    paths = {"1": (1, 93), "2": (2, 93), "10": (1, 0, 93)}
    nodes = {(): {1: math.log(.6), 2: math.log(.4)},
             (1,): {93: math.log(.1), 0: math.log(.9)},
             (2,): {93: math.log(.9)}, (1, 0): {93: math.log(.9)}}
    calls = []

    def next_probs(tokens, required):
        node = tuple(tokens[1:])
        calls.append(node)
        assert required <= nodes[node].keys()
        return nodes[node]

    result = score_trie([999], paths, next_probs)
    joint = {"1": .6 * .1, "2": .4 * .9, "10": .6 * .9 * .9}
    total = sum(joint.values())
    assert result["probabilities"] == pytest.approx({k: p / total for k, p in joint.items()})
    assert result["probabilities"]["2"] > result["probabilities"]["1"]
    assert len(calls) == len(set(calls)) == result["trie_node_count"] == 4


def test_prefix_boundaries_checked():
    prefix, paths = suffix_paths("[", ["1", "10"], lambda s: list(map(ord, s)))
    assert prefix == [91]
    assert paths["10"] == (49, 48, 93)
    with pytest.raises(ScoringUnavailable, match="prefix"):
        suffix_paths("[", ["1"], lambda s: [len(s)])


@pytest.mark.parametrize("payload", [
    {}, {"probs": [{"probs": [{"tok_str": "1", "prob": .9}]}]},
    {"probs": [{"top_logprobs": [{"token": "1", "logprob": -.1}]}]},
    {"truncated": True},
])
def test_missing_raw_token_scores_fail(payload):
    with pytest.raises(ScoringUnavailable):
        extract_logprobs(payload)


def test_fake_http_scorer_uses_template_and_unmodified_probabilities():
    completion_calls = []
    templates = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/apply-template":
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            templates.append(body)
            return httpx.Response(200, json={"prompt": "assistant:"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(map(ord, body["content"]))})
        assert request.url.path == "/completion"
        completion_calls.append(body)
        assert body["temperature"] == -1
        assert body["n_predict"] == 1
        assert body["post_sampling_probs"] is False
        assert "grammar" not in body and "logit_bias" not in body
        return httpx.Response(200, json={"completion_probabilities": [{"top_logprobs": [
            {"id": token, "logprob": -1.0} for token in (49, 50, 93)]}]})

    client = ModelClient("http://fake/v1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    scorer = JevScorer(client, Taxonomy({"1": "network", "2": "audio"}), "{options}\nQuery: {query}")
    result = scorer.score("literal {options}")
    assert result["probabilities"] == pytest.approx({"1": .5, "2": .5})
    assert result["prompt"].endswith("Best answer: [")
    assert templates[0]["messages"][0]["content"].endswith("literal {options}")
    assert len(completion_calls) == 3
    client.close()


def test_missing_candidate_is_not_assigned_zero():
    class Fake:
        def render(self, messages):
            return "p"

        def tokenize(self, text):
            return list(map(ord, text))

        def post(self, path, body):
            return {"probs": [{"top_logprobs": [{"id": 49, "logprob": -1}]}]}

    with pytest.raises(ScoringUnavailable, match="candidate outside"):
        JevScorer(Fake(), Taxonomy({"1": "one", "2": "two"}), "{query}", 2, 4).score("test")
