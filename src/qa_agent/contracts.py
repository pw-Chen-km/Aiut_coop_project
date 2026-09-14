"""Small dependency-injection interfaces; implementations remain interchangeable."""
from __future__ import annotations

from typing import Protocol


class Router(Protocol):
    def route(self, question: str, *, qid: str = "interactive", feedback: dict | None = None) -> dict: ...


class Retriever(Protocol):
    def search(self, question: str, scope: dict, *, k: int) -> list[dict]: ...


class Generator(Protocol):
    def generate(self, question: str, items: list[dict], *, qid: str = "interactive") -> dict: ...
