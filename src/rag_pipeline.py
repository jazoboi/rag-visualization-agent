"""
End-to-end RAG pipeline for document Q&A with visualization.

Orchestrates document ingestion, embedding, retrieval, and
LLM-based answer generation with optional chart output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from a RAG query."""
    answer: str
    sources: list[dict[str, Any]]
    chart_config: dict | None = None
    retrieval_scores: list[float] | None = None


class RAGPipeline:
    """Orchestrates retrieval-augmented generation with visualization.

    Parameters
    ----------
    vector_store : VectorStore
        Configured ChromaDB vector store.
    llm_client : OpenAI
        OpenAI client for generation.
    top_k : int
        Number of chunks to retrieve (default: 5).
    """

    SYSTEM_PROMPT = (
        "You are an analyst. Answer questions using ONLY the provided context. "
        "If the answer contains numeric data, include a JSON chart config at the end "
        "with format: ```chart\n{type, data, title}\n```"
    )

    def __init__(self, vector_store, llm_client: OpenAI, top_k: int = 5) -> None:
        self._store = vector_store
        self._llm = llm_client
        self._top_k = top_k

    def query(self, question: str) -> RAGResponse:
        """Execute a RAG query with optional visualization.

        Parameters
        ----------
        question : str
            User's natural language question.

        Returns
        -------
        RAGResponse
            Answer with sources and optional chart configuration.
        """
        # Retrieve relevant chunks
        results = self._store.search(question, top_k=self._top_k)
        context = "\n\n---\n\n".join(
            r["text"] for r in results
        )

        # Generate answer
        response = self._llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        chart_config = self._extract_chart_config(answer)

        return RAGResponse(
            answer=answer.split("```chart")[0].strip(),
            sources=results,
            chart_config=chart_config,
            retrieval_scores=[r.get("score", 0) for r in results],
        )

    @staticmethod
    def _extract_chart_config(text: str) -> dict | None:
        """Extract chart configuration from LLM response."""
        if "```chart" not in text:
            return None
        try:
            import json
            chart_block = text.split("```chart")[1].split("```")[0].strip()
            return json.loads(chart_block)
        except (IndexError, json.JSONDecodeError):
            return None

    def ingest_documents(self, documents: list[dict]) -> int:
        """Ingest documents into the vector store.

        Parameters
        ----------
        documents : list[dict]
            List of dicts with 'text' and 'metadata' keys.

        Returns
        -------
        int
            Number of chunks stored.
        """
        count = 0
        for doc in documents:
            chunks = self._chunk_text(doc["text"])
            for chunk in chunks:
                self._store.add(chunk, metadata=doc.get("metadata", {}))
                count += 1
        logger.info("Ingested %d chunks from %d documents", count, len(documents))
        return count

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
