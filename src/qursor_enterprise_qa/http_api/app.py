"""Stateless REST API for the local evidence-grounded QA runtime."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from qursor_enterprise_qa.answer_generation import AnswerGenerator, QueryRewriter
from qursor_enterprise_qa.corpus_runtime import CorpusBundle
from qursor_enterprise_qa.history_context import (
    HistoryPair,
    build_answer_history,
    build_retrieval_query,
    build_rewrite_history,
    has_query_rewrite_trigger,
)
from qursor_enterprise_qa.settings import Settings


NON_ANSWER_TEXT = "I can't answer this"
CLARIFY_TEXT = "Which Qursor system, document, or procedure are you referring to?"


class HistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Q: str = Field(min_length=1, max_length=20000)
    A: str = Field(min_length=1, max_length=20000)

    @field_validator("Q", "A")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: int = Field(strict=True)
    context: list[HistoryItem]
    question: str = Field(min_length=1, max_length=20000)

    @field_validator("question")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class QAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: int
    response_type: Literal["answer", "clarify", "abstain"]
    answer: str
    sources: list[dict[str, Any]]


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    """Expose only the citation fields required by API consumers."""
    page_start = source.get("page_start")
    page_end = source.get("page_end")
    pages: list[int] = []
    if isinstance(page_start, int):
        pages = (
            list(range(page_start, page_end + 1))
            if isinstance(page_end, int) and page_end >= page_start
            else [page_start]
        )
    return {
        "document": source.get("document_title") or source.get("filename") or source.get("document_id"),
        "pages": pages,
        "excerpt": source.get("excerpt", ""),
    }


@dataclass(frozen=True)
class PipelineTrace:
    """Internal timings used by the offline benchmark, never returned by API."""

    rewrite_triggered: bool
    rewrite_succeeded: bool
    rewrite_query: str | None
    rewrite_ms: float
    retrieval_ms: float
    answer_ms: float
    total_ms: float
    rewrite_qwen_calls: int
    answer_qwen_calls: int
    retrieved_documents: tuple[str, ...]
    response_type: str
    answerable: bool
    generated_answer: str
    retrieved_chunks: tuple[dict[str, Any], ...]
    final_k: int


def _call_count(component: Any) -> int:
    return int(getattr(component, "call_count", 0))


def _clarify_response(
    body: QARequest,
    *,
    total_started: float,
    rewrite_triggered: bool,
    rewrite_query: str | None,
    rewrite_ms: float,
    rewrite_qwen_calls: int,
    final_k: int,
) -> tuple[dict[str, Any], PipelineTrace]:
    """Return a deterministic clarification without retrieval or answer LLM."""
    response = {
        "conversation_id": body.conversation_id,
        "response_type": "clarify",
        "answer": CLARIFY_TEXT,
        "sources": [],
    }
    trace = PipelineTrace(
        rewrite_triggered=rewrite_triggered,
        rewrite_succeeded=bool(rewrite_query),
        rewrite_query=rewrite_query,
        rewrite_ms=rewrite_ms,
        retrieval_ms=0.0,
        answer_ms=0.0,
        total_ms=(time.perf_counter() - total_started) * 1000,
        rewrite_qwen_calls=rewrite_qwen_calls,
        answer_qwen_calls=0,
        retrieved_documents=(),
        response_type="clarify",
        answerable=False,
        generated_answer=CLARIFY_TEXT,
        retrieved_chunks=(),
        final_k=final_k,
    )
    return response, trace


def run_question(
    body: QARequest,
    *,
    config: Settings,
    bundle: CorpusBundle,
    generator: AnswerGenerator,
    rewriter: QueryRewriter | None,
    final_k: int = 5,
) -> tuple[dict[str, Any], PipelineTrace]:
    """Run one request and return its public response plus internal trace."""
    if final_k < 1 or final_k > 20:
        raise ValueError("final_k must be between 1 and 20")
    total_started = time.perf_counter()
    # Exclude the entire abstained turn before rewrite, retrieval and answer budgets.
    # Context has no response_type yet, so match only our exact public fallback.
    history = [
        HistoryPair(item.Q, item.A)
        for item in body.context
        if item.A.strip().casefold() != NON_ANSWER_TEXT.casefold()
    ]
    rewrite_requested = (
        config.query_rewrite_mode == "conditional"
        and has_query_rewrite_trigger(body.question)
    )
    rewrite_triggered = rewrite_requested and bool(history)
    rewrite_query: str | None = None
    rewrite_started = time.perf_counter()
    rewrite_calls_before = _call_count(rewriter) if rewriter is not None else 0
    if rewrite_requested and not history:
        return _clarify_response(
            body,
            total_started=total_started,
            rewrite_triggered=True,
            rewrite_query=None,
            rewrite_ms=0.0,
            rewrite_qwen_calls=0,
            final_k=final_k,
        )
    if rewrite_triggered and rewriter is not None:
        rewrite_query = rewriter.rewrite(
            body.question,
            build_rewrite_history(history, config.retrieval_history_chars),
        )
    rewrite_ms = (time.perf_counter() - rewrite_started) * 1000
    rewrite_qwen_calls = (
        _call_count(rewriter) - rewrite_calls_before if rewriter is not None else 0
    )
    if rewrite_triggered and not rewrite_query:
        return _clarify_response(
            body,
            total_started=total_started,
            rewrite_triggered=True,
            rewrite_query=None,
            rewrite_ms=rewrite_ms,
            rewrite_qwen_calls=rewrite_qwen_calls,
            final_k=final_k,
        )

    retrieval_started = time.perf_counter()
    if config.query_rewrite_mode == "disabled":
        retrieval_query = build_retrieval_query(
            history, body.question, config.retrieval_history_chars
        )
        results = bundle.search(retrieval_query, candidate_k=20, final_k=final_k)
    elif rewrite_query:
        results = bundle.search_with_rewrite(
            body.question, rewrite_query, candidate_k=20, final_k=final_k
        )
    else:
        results = bundle.search(body.question, candidate_k=20, final_k=final_k)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    model_history = build_answer_history(history, config.answer_history_chars)
    answer_started = time.perf_counter()
    answer_calls_before = _call_count(generator)
    generated = generator.generate(body.question, model_history, results)
    answer_ms = (time.perf_counter() - answer_started) * 1000
    answer_qwen_calls = _call_count(generator) - answer_calls_before

    document_records = getattr(bundle, "documents", {})
    retrieved_documents = tuple(
        document_records.get(result.doc_id, {}).get("title", result.doc_id)
        for result in results
    )

    response_type = generated["response_type"]
    by_citation = {f"C{index:03d}": result for index, result in enumerate(results, 1)}
    # Reorder only cited passages; preserve retrieval rank when scores tie.
    cited_ids = sorted(
        generated["citation_ids"],
        key=lambda citation_id: (
            -by_citation[citation_id].score,
            by_citation[citation_id].rank,
        ),
    ) if response_type == "answer" else []
    sources = [
        bundle.source(by_citation[citation_id], citation_id)
        for citation_id in cited_ids
    ]
    answer = (
        NON_ANSWER_TEXT
        if response_type == "abstain"
        else generated["answer"]
    )
    response = {
        "conversation_id": body.conversation_id,
        "response_type": response_type,
        "answer": answer,
        "sources": [_public_source(source) for source in sources],
    }
    trace = PipelineTrace(
        rewrite_triggered=rewrite_triggered,
        rewrite_succeeded=bool(rewrite_query),
        rewrite_query=rewrite_query,
        rewrite_ms=rewrite_ms,
        retrieval_ms=retrieval_ms,
        answer_ms=answer_ms,
        total_ms=(time.perf_counter() - total_started) * 1000,
        rewrite_qwen_calls=rewrite_qwen_calls,
        answer_qwen_calls=answer_qwen_calls,
        retrieved_documents=retrieved_documents,
        response_type=str(generated["response_type"]),
        answerable=bool(generated["answerable"]),
        generated_answer=str(generated["answer"]),
        retrieved_chunks=tuple(
            {
                "rank": result.rank,
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "score": result.score,
                "component_scores": result.component_scores,
                "text": result.text,
                "source_spans": result.metadata.get("source_spans", []),
            }
            for result in results
        ),
        final_k=final_k,
    )
    return response, trace


def create_app(
    *,
    settings: Settings | None = None,
    bundle: CorpusBundle | None = None,
    generator: AnswerGenerator | None = None,
    rewriter: QueryRewriter | None = None,
) -> FastAPI:
    config = settings or Settings.from_env()
    state: dict[str, Any] = {
        "bundle": bundle,
        "generator": generator,
        "rewriter": rewriter,
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["bundle"] is None:
            state["bundle"] = CorpusBundle(config.corpus_dir, model_dir=config.e5_model_dir)
        if state["generator"] is None:
            state["generator"] = AnswerGenerator(
                config.qwen_endpoint, config.qwen_model,
                timeout=config.qwen_timeout_seconds, max_tokens=config.qwen_max_tokens,
            )
        if config.query_rewrite_mode == "conditional" and state["rewriter"] is None:
            state["rewriter"] = QueryRewriter(
                config.qwen_endpoint,
                config.qwen_model,
                timeout=config.query_rewrite_timeout_seconds,
                max_tokens=config.query_rewrite_max_tokens,
            )
        yield
        if bundle is None and state["bundle"] is not None:
            state["bundle"].close()

    app = FastAPI(title="Qursor Enterprise QA", version="0.1.0", lifespan=lifespan)

    def resources() -> tuple[CorpusBundle, AnswerGenerator]:
        if state["bundle"] is None or state["generator"] is None:
            raise HTTPException(status_code=503, detail="service is not ready")
        return state["bundle"], state["generator"]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        try:
            bundle_obj, generator_obj = resources()
            generator_obj.ping()
            return {"status": "ready", "corpus_version": bundle_obj.version}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"not ready: {type(exc).__name__}") from exc

    def execute_question(
        body: QARequest,
        *,
        final_k: int,
        error_label: str,
    ) -> dict[str, Any]:
        bundle_obj, generator_obj = resources()
        try:
            response, _trace = run_question(
                body,
                config=config,
                bundle=bundle_obj,
                generator=generator_obj,
                rewriter=state["rewriter"],
                final_k=final_k,
            )
            return response
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"{error_label} failed: {type(exc).__name__}",
            ) from exc

    @app.post("/v1/qa", response_model=QAResponse)
    def answer_question(body: QARequest) -> dict[str, Any]:
        return execute_question(body, final_k=5, error_label="QA request")

    @app.post("/internal/ablation/qa", response_model=QAResponse)
    def ablation_question(
        body: QARequest,
        final_k: int = Query(..., ge=1, le=20),
    ) -> dict[str, Any]:
        return execute_question(body, final_k=final_k, error_label="ablation request")

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("qursor_enterprise_qa.http_api.app:app", host="0.0.0.0", port=8000)
