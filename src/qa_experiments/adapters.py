"""Adapters retain native upstream algorithms; only runtime/IO are substituted."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import types

from .transport import LocalTransport, RecordingOpener, ToolError

def read(path):
    return json.loads(Path(path).read_text())


class ExistingQA:
    def __init__(self, root, method, transport):
        from qa_agent.factory import create_session
        config = read(Path(root) / "online.json")
        self.session = create_session(config)
        self.agent = self.session.agent
        self.method, self.transport = method, transport
        self.agent.router.client._opener = RecordingOpener(transport)
        self.agent.generator.client._opener = RecordingOpener(transport)

    def answer(self, question):
        if self.method == "navigation":
            result = self.agent.answer(question["question"], qid=question["qid"])
            result["all_context_items"] = [item for r in result.get("rounds", []) for item in r.get("context_items", [])]
            result["all_items"] = [item for r in result.get("rounds", []) for item in r.get("items", [])]
            return result
        from qa_agent.retrieval import GlobalDenseRetriever
        retriever = GlobalDenseRetriever(self.session.store, self.agent.retriever.encoder)
        items = retriever.search(question["question"], k=20)
        result = self.agent.generator.generate(question["question"], items, qid=question["qid"])
        result.update(items=items, stop_reason="answered" if result.get("answerable") else "insufficient_context")
        return result

    def close(self):
        self.session.close()


class ARAG:
    def __init__(self, root, environment, transport, *, build=False, max_loops=15):
        root = Path(root)
        sys.path.insert(0, str(Path(environment["repo"]) / "src"))
        self.root, self.transport = root, transport
        self.max_loops = max_loops
        from arag.agent.base import BaseAgent
        from arag.tools.keyword_search import KeywordSearchTool
        from arag.tools.semantic_search import SemanticSearchTool
        from arag.tools.read_chunk import ReadChunkTool
        from arag.tools.registry import ToolRegistry
        index = root / "indexes/arag"
        if build:
            if index.exists():
                raise FileExistsError("Do not rebuild an existing index")
            spec = importlib.util.spec_from_file_location("native_arag_index", Path(environment["repo"]) / "scripts/build_index.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.build_index(str(root / "chunks.json"), str(index), environment["model"], device="cpu", batch_size=8)
        if not (index / "sentence_index.pkl").exists():
            raise FileNotFoundError("Required semantic index missing; no lexical-only fallback")
        self.chunks = {r["id"]: r for r in read(root / "chunks.json")}
        registry = ToolRegistry()
        counter = types.SimpleNamespace(encode=lambda text: range(transport.count_text(text)))
        # Constructors otherwise download/use GPT tokenization. This isolated
        # process uses the real serving tokenizer for all native accounting.
        import tiktoken
        tiktoken.encoding_for_model = lambda _: counter
        for tool in (KeywordSearchTool(str(root / "chunks.json")),
                     SemanticSearchTool(str(root / "chunks.json"), str(index), environment["model"], "cpu"),
                     ReadChunkTool(str(root / "chunks.json"))):
            tool.tokenizer = counter
            registry.register(tool)
        native_execute = registry.execute
        self.tool_checks = []
        def execute(name, context, **kwargs):
            if name == "read_chunk" and any(cid not in self.chunks for cid in kwargs.get("chunk_ids", [])):
                self.tool_checks.append({"status": "invalid_tool_call", "error": "Unknown read_chunk ID"})
            result, log = native_execute(name, context, **kwargs)
            if log.get("error"):
                self.tool_checks.append({"status": "tool_execution_error", "error": str(log["error"])})
            return result, log
        registry.execute = execute
        self.registry = registry
        self.prompt = (Path(environment["repo"]) / "src/arag/agent/prompts/default.txt").read_text()
        self.base_class = BaseAgent
        self.counter = counter

    def capability_check(self):
        from arag.core.context import AgentContext
        context = AgentContext()
        first = next(iter(self.chunks.values()))
        word = next(w for w in re.findall(r"[A-Za-z]{4,}", first["text"]))
        calls = [("keyword_search", {"keywords": [word]}),
                 ("semantic_search", {"query": word}), ("read_chunk", {"chunk_ids": [first["id"]]})]
        result = {}
        for name, args in calls:
            text, log = self.registry.execute(name, context, **args)
            if log.get("error") or not text or "No results" in text:
                raise RuntimeError("Native tool capability check failed: " + name)
            result[name] = {"ok": True, "log": log}
        return result

    def answer(self, question):
        self.tool_checks = []
        agent = self.base_class(self.transport, self.registry, self.prompt, max_loops=self.max_loops,
                                max_token_budget=7000, verbose=False)
        agent.tokenizer = self.counter
        # Include the actual Qwen template and native tool schemas in budgeting.
        agent._calculate_message_tokens = lambda messages: self.transport.count_request(
            {"messages": messages, "tools": self.registry.get_all_schemas(), "tool_choice": "auto"})
        result = agent.run(question["question"])
        result["tool_checks"] = self.tool_checks
        ids = []
        contexts = []
        for step in result.get("trajectory", []):
            aliases = re.findall(r"Chunk ID: (\d+)|\[Chunk (\d+)\]", step.get("tool_result", ""))
            selected = list(dict.fromkeys(a or b for a,b in aliases))
            ids.extend(self.chunks[a]["chunk_id"] for a in selected if a in self.chunks)
            # These are observed tool returns, not the full unseen parent chunks.
            contexts.append({"content_type": "tool_result", "tool": step["tool_name"],
                "text": step.get("tool_result", ""), "chunk_ids": [self.chunks[a]["chunk_id"] for a in selected if a in self.chunks]})
        result.update(status="ok", context_items=contexts, retrieved_chunk_ids=list(dict.fromkeys(ids)),
                      stop_reason=("loop_limit" if result.get("max_loops_exceeded") else
                                   "budget_limit" if result.get("token_budget_exceeded") else "answered"))
        return result

    def close(self):
        pass


class Linear:
    def __init__(self, root, environment, transport, settings, *, build=False):
        sys.path.insert(0, environment["repo"])
        from src.LinearRAG import LinearRAG
        from src.config import LinearRAGConfig
        from sentence_transformers import SentenceTransformer
        root = Path(root)
        self.transport = transport
        self.chunks = read(root / "chunks.json")
        self.by_text = {f'{i}:{c["text"]}': c for i,c in enumerate(self.chunks)}
        work = root / "indexes/linear"
        if not build and not (work / "qursor/LinearRAG.graphml").exists():
            raise FileNotFoundError("LinearRAG index not built")
        if build and work.exists():
            raise FileExistsError("Do not overwrite an existing LinearRAG index")
        class ReadOnlyEncoder:
            def __init__(self, model):
                self.model = model
            def encode(self, texts, **kwargs):
                values = [texts] if isinstance(texts, str) else texts
                maximum = self.model.max_seq_length
                if any(len(self.model.tokenizer.encode(t)) > maximum for t in values):
                    raise ValueError("Native embedding would truncate a corpus item")
                return self.model.encode(texts, **kwargs)
        embedding = SentenceTransformer(environment["model"], device="cpu")
        config = LinearRAGConfig(dataset_name="qursor", embedding_model=ReadOnlyEncoder(embedding),
            llm_model=types.SimpleNamespace(infer=lambda messages: transport.chat(messages)["message"]["content"]),
            working_dir=str(work), spacy_model="en_core_web_trf", max_workers=1, batch_size=16,
            use_vectorized_retrieval=False, **settings)
        self.rag = LinearRAG(config)
        original_chunks = self.chunks
        def within_document_adjacency(this):
            indexed = {}
            for key, text in this.passage_embedding_store.get_hash_id_to_text().items():
                match = re.match(r"^(\d+):", text)
                if match:
                    indexed[int(match.group(1))] = key
            for i in range(len(original_chunks)-1):
                if original_chunks[i]["doc_id"] == original_chunks[i+1]["doc_id"]:
                    this.node_to_node_stats[indexed[i]][indexed[i+1]] = 1.0
        self.rag.add_adjacent_passage_edges = types.MethodType(within_document_adjacency, self.rag)
        if build:
            self.rag.index(list(self.by_text))
        else:
            # Reconstruct in-memory graph from the pinned cache without invoking index()
            # (upstream index rewrites cache/graph files even on cache hits).
            import igraph as ig
            self.rag.graph = ig.Graph.Read_GraphML(str(work / "qursor/LinearRAG.graphml"))
            passage_ids = set(self.rag.passage_embedding_store.hash_ids)
            self.rag.passage_node_indices = [v.index for v in self.rag.graph.vs if v["name"] in passage_ids]
            passages, sentences, _ = self.rag.load_existing_data(self.rag.passage_embedding_store.hash_ids)
            _, _, _, e2s, s2e = self.rag.extract_nodes_and_edges(passages, sentences)
            self.rag.entity_hash_id_to_sentence_hash_ids = {self.rag.entity_embedding_store.text_to_hash_id[e]:
                [self.rag.sentence_embedding_store.text_to_hash_id[s] for s in values] for e,values in e2s.items()}
            self.rag.sentence_hash_id_to_entity_hash_ids = {self.rag.sentence_embedding_store.text_to_hash_id[s]:
                [self.rag.entity_embedding_store.text_to_hash_id[e] for e in values] for s,values in s2e.items()}

    def answer(self, question):
        no_entities = not self.rag.spacy_ner.question_ner(question["question"])
        # Upstream retrieve() requires this key only to copy it to its output.
        # Never provide benchmark GT: the reader prompt uses passages/question.
        result = self.rag.qa([{"qid": question["qid"], "question": question["question"], "answer": ""}])[0]
        result.pop("gold_answer", None)
        contexts = [dict(self.by_text[text]) for text in result["sorted_passage"]]
        result.update(answer=result.pop("pred_answer"), status="ok", context_items=contexts,
            retrieved_chunk_ids=[c["chunk_id"] for c in contexts], stop_reason="answered",
            native_dense_fallback=no_entities)
        return result

    def close(self):
        pass
