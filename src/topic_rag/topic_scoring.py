"""Training-free joint candidate-suffix scoring (LLM-as-Jev Eq. 1).

Uses a shared prompt and a trie over full suffix token sequences (including ]).
HTTP transport evaluates trie nodes sequentially; it does not claim the
paper's batched shared-KV latency. No grammar/logit bias alters probabilities.
"""
import math
import time


class ScoringUnavailable(RuntimeError):
    pass


def softmax(values):
    if not values or any(not math.isfinite(v) for v in values.values()):
        raise ScoringUnavailable("nonfinite/missing candidate log likelihood")
    maximum = max(values.values())
    weights = {k: math.exp(v - maximum) for k, v in values.items()}
    total = math.fsum(weights.values())
    return {k: v / total for k, v in weights.items()}


def suffix_paths(prompt, options, tokenize):
    prefix = tokenize(prompt)
    paths = {}
    for option in options:
        full = tokenize(prompt + option + "]")
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ScoringUnavailable("candidate tokenization changes prompt prefix")
        paths[option] = tuple(full[len(prefix):])
    if len(set(paths.values())) != len(paths):
        raise ScoringUnavailable("candidate suffix collision")
    return prefix, paths


def score_trie(prefix, paths, next_logprobs):
    # A node is queried once even when multiple options share its prefix.
    cache, traces = {}, {}
    scores = {}
    for option, path in paths.items():
        terms = []
        for position, token in enumerate(path):
            node = path[:position]
            if node not in cache:
                required = {p[position] for p in paths.values()
                            if len(p) > position and p[:position] == node}
                cache[node] = next_logprobs(prefix + list(node), required)
            value = cache[node].get(token)
            if value is None or not math.isfinite(value) or value > 1e-6:
                raise ScoringUnavailable(f"missing/invalid probability for token {token}")
            terms.append(value)
        scores[option] = math.fsum(terms)
        traces[option] = {"token_ids": list(path), "token_logprobs": terms}
    return {"probabilities": softmax(scores), "candidate_log_likelihoods": scores,
            "candidate_tokens": traces, "trie_node_count": len(cache)}


def extract_logprobs(payload):
    if payload.get("truncated"):
        raise ScoringUnavailable("model truncated scoring prompt")
    entries = payload.get("completion_probabilities", payload.get("probs", []))
    if not entries or not isinstance(entries[0], dict):
        raise ScoringUnavailable("server did not return token log probabilities")
    top = entries[0].get("top_logprobs")
    if not isinstance(top, list):
        # Do not accept post-sampling probabilities or approximate missing tokens.
        raise ScoringUnavailable("raw top_logprobs with token IDs are required")
    result = {}
    for item in top:
        if type(item.get("id")) is not int or "logprob" not in item:
            raise ScoringUnavailable("probability entries require token IDs and logprob")
        result[item["id"]] = float(item["logprob"])
    return result


class JevScorer:
    def __init__(self, model, taxonomy, prompt_template, n_probs=256, max_n_probs=1024):
        if not 1 <= n_probs <= max_n_probs:
            raise ValueError("require 1 <= n_probs <= max_n_probs")
        self.model, self.taxonomy, self.template = model, taxonomy, prompt_template
        self.n_probs, self.max_n_probs = n_probs, max_n_probs

    def _next(self, tokens, required):
        n_probs = self.n_probs
        while True:
            payload = self.model.post("/completion", {
                "prompt": tokens, "n_predict": 1, "temperature": -1,
                "n_probs": n_probs, "post_sampling_probs": False,
                "stream": False, "cache_prompt": True, "seed": 0,
            })
            self.requests += 1
            scores = extract_logprobs(payload)
            if required <= scores.keys():
                return scores
            if n_probs >= self.max_n_probs:
                raise ScoringUnavailable("candidate outside returned logits; increase max_n_probs or use local logits")
            n_probs = min(self.max_n_probs, n_probs * 2)

    def score(self, query):
        started = time.perf_counter()
        self.requests = 0
        options = "\n".join(f"[{k}] {v}" for k, v in self.taxonomy.options.items())
        # format substitutes once; braces inside the query remain literal data.
        user = self.template.format(query=query, options=options)
        prompt = self.model.render([{"role": "user", "content": user}]) + "Best answer: ["
        prefix, paths = suffix_paths(prompt, self.taxonomy.options, self.model.tokenize)
        result = score_trie(prefix, paths, self._next)
        result.update({
            "selected_option": max(result["probabilities"], key=result["probabilities"].get),
            "prompt": prompt, "prompt_tokens": len(prefix), "completion_requests": self.requests,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "method": "joint_full_suffix_loglikelihood_sequential_http_trie",
        })
        return result

