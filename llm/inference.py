import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_GENERATION_MODEL = "gpt-5.4-nano"
DEFAULT_MAX_OUTPUT_TOKENS = 300


@dataclass
class SourceSnippet:
    text: str
    source_file: str = ""
    page: int = 0
    chunk_id: str = ""
    score: Optional[float] = None


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)


class LLMConfigurationError(RuntimeError):
    pass


def create_openai_client(api_key: Optional[str] = None) -> Any:
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise LLMConfigurationError("OPENAI_API_KEY is required")
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=resolved_key)


def build_response_input(prompt: str, sources: Iterable[SourceSnippet]) -> List[Dict[str, str]]:
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        label_parts = []
        if source.source_file:
            label_parts.append(source.source_file)
        if source.page:
            label_parts.append(f"page {source.page}")
        if source.chunk_id:
            label_parts.append(f"chunk {source.chunk_id}")
        label = ", ".join(label_parts) or f"source {index}"
        context_blocks.append(f"[Source {index}: {label}]\n{source.text.strip()}")

    if context_blocks:
        context = "\n\n".join(context_blocks)
        user_content = (
            "Answer the question using the retrieved textbook context below. "
            "If the context is insufficient, say what is missing instead of inventing details.\n\n"
            f"Question:\n{prompt.strip()}\n\n"
            f"Retrieved context:\n{context}"
        )
    else:
        user_content = prompt.strip()

    return [
        {
            "role": "system",
            "content": (
                "You are a concise distributed-systems teaching assistant. "
                "Use only the provided context when context is supplied."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {}


def _extract_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    # Defensive fallback for SDK versions that expose the raw output tree.
    pieces: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()


async def generate_answer(
    prompt: str,
    sources: Iterable[SourceSnippet],
    *,
    client: Any,
    model: str = DEFAULT_GENERATION_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> LLMResult:
    response = await client.responses.create(
        model=model,
        input=build_response_input(prompt, sources),
        max_output_tokens=max_output_tokens,
    )
    text = _extract_text(response)
    if not text:
        raise RuntimeError("OpenAI returned an empty response")
    return LLMResult(text=text, model=model, usage=_usage_to_dict(getattr(response, "usage", None)))


def run_llm(query: str, context: str) -> str:
    raise RuntimeError(
        "run_llm() has been replaced by async generate_answer(); use the worker HTTP service instead"
    )
