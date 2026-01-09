"""Answer generation with optional LLM."""

from backend.config import get_settings
from .retriever import RetrievalResult


class AnswerGenerator:
    """Generates answers from retrieved context."""

    def __init__(self):
        self._client = None
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize LLM client if API key available."""
        settings = get_settings()
        if settings.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=settings.openai_api_key)
            except ImportError:
                pass

    def generate(
        self,
        question: str,
        context: list[RetrievalResult],
        max_tokens: int = 500,
    ) -> dict:
        """Generate an answer from the question and context.

        Args:
            question: User's question.
            context: Retrieved context chunks.
            max_tokens: Maximum tokens in response.

        Returns:
            Dict with 'answer', 'sources', and 'method'.
        """
        if not context:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "method": "no_context",
            }

        # Build context string
        context_text = self._format_context(context)
        sources = self._extract_sources(context)

        # If no LLM, return raw excerpts
        if self._client is None:
            return {
                "answer": self._fallback_answer(question, context),
                "sources": sources,
                "method": "excerpt",
            }

        # Generate with LLM
        answer = self._generate_with_llm(question, context_text, max_tokens)
        return {
            "answer": answer,
            "sources": sources,
            "method": "llm",
        }

    def _format_context(self, context: list[RetrievalResult]) -> str:
        """Format context for LLM prompt."""
        parts = []
        for i, result in enumerate(context, 1):
            source = result.metadata.get("section_title", result.document_id)
            parts.append(f"[{i}] {source}:\n{result.text}\n")
        return "\n".join(parts)

    def _extract_sources(self, context: list[RetrievalResult]) -> list[dict]:
        """Extract source citations from context."""
        sources = []
        for result in context:
            sources.append(
                {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "section": result.metadata.get("section_title"),
                    "score": round(result.score, 3),
                }
            )
        return sources

    def _fallback_answer(
        self, question: str, context: list[RetrievalResult]
    ) -> str:
        """Generate answer from excerpts when no LLM available."""
        excerpts = []
        for result in context[:3]:  # Top 3 results
            text = result.text[:300]
            if len(result.text) > 300:
                text += "..."
            source = result.metadata.get("section_title", result.document_id)
            excerpts.append(f"**{source}**:\n{text}")

        return "Relevant excerpts:\n\n" + "\n\n---\n\n".join(excerpts)

    def _generate_with_llm(
        self, question: str, context: str, max_tokens: int
    ) -> str:
        """Generate answer using LLM."""
        system_prompt = """You are a regulatory expert assistant. Answer questions based solely on the provided context.
If the context doesn't contain enough information, say so.
Always cite the source when making claims.
Be precise and concise."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
