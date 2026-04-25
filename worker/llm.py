from __future__ import annotations

import asyncio
import os

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError


SYSTEM_TEMPLATE = """You are a helpful assistant.
Use the following retrieved context to answer the user's question accurately.
If the context is not relevant, answer from your own knowledge.

Context:
{context}
"""

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI | None:
    global _client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def _mock_response(query: str, context: str) -> str:
    preview = context.replace("\n", " ")[:240]
    return (
        "OpenAI is not configured, so this worker returned a local demo response. "
        f"Query: {query}. Retrieved context preview: {preview}"
    )


async def run_llm(query: str, context: str) -> str:
    if os.environ.get("MOCK_LLM", "").lower() in {"1", "true", "yes"}:
        return _mock_response(query, context)

    client = _get_client()
    if client is None:
        return _mock_response(query, context)

    retryable_errors = (APIConnectionError, APITimeoutError, APIError, RateLimitError)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "300")),
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_TEMPLATE.format(context=context),
                    },
                    {"role": "user", "content": query},
                ],
            )
            return response.choices[0].message.content or ""
        except retryable_errors as exc:
            last_error = exc
            await asyncio.sleep(0.5 * (2**attempt))

    return f"LLM call failed after retries: {last_error}"
