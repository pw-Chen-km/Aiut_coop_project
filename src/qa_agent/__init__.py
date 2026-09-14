"""Online, local-only QA: navigation -> scoped retrieval -> grounded answer.

Importing this package does not load a model, parse a PDF, or open a database.
"""

from .pipeline import QAAgent

__all__ = ["QAAgent"]
