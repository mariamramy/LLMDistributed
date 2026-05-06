import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import aiohttp

DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_GENERATION_MODEL = "llama3.2:3b-instruct-q3_K_M"
DEFAULT_MAX_OUTPUT_TOKENS = 300
DEFAULT_TEMPERATURE = 0.2
DEFAULT_EMBED_MAX_CHARS = 200


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


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, timeout_s: Optional[float] = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip(
            "/"
        )
        self.timeout_s = float(timeout_s or os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "120"))

    async def _get(self, path: str) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}{path}") as response:
                return await _json_or_error(response)

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}{path}", json=payload) as response:
                return await _json_or_error(response)

    async def list_models(self) -> List[str]:
        response = await self._get("/api/tags")
        models = response.get("models", [])
        return [str(model.get("name", "")) for model in models if model.get("name")]

    async def has_model(self, model: str) -> bool:
        installed_models = await self.list_models()
        return any(_model_matches(model, installed) for installed in installed_models)

    async def models_ready(self, generation_model: str, embedding_model: str) -> bool:
        installed_models = await self.list_models()
        return any(_model_matches(generation_model, model) for model in installed_models) and any(
            _model_matches(embedding_model, model) for model in installed_models
        )

    async def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        return await self._post(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "messages": messages,
                "options": {
                    "num_predict": max_output_tokens,
                    "temperature": temperature,
                },
            },
        )

    async def embed_texts(self, texts: Iterable[str], *, model: str) -> List[List[float]]:
        input_texts = [_trim_embedding_text(text) for text in texts]
        response = await self._post(
            "/api/embed",
            {
                "model": model,
                "input": input_texts,
                "truncate": True,
            },
        )
        embeddings = response.get("embeddings")
        if embeddings is None:
            single_embedding = response.get("embedding")
            embeddings = [single_embedding] if single_embedding is not None else None
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama returned no embeddings")
        return embeddings


def create_ollama_client(base_url: Optional[str] = None) -> OllamaClient:
    return OllamaClient(base_url=base_url)


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


async def _json_or_error(response: aiohttp.ClientResponse) -> Dict[str, Any]:
    if response.status >= 400:
        body = await response.text()
        raise RuntimeError(f"Ollama request failed with HTTP {response.status}: {body}")
    return await response.json()


def _model_matches(requested: str, installed: str) -> bool:
    if requested == installed:
        return True
    if ":" not in requested and installed == f"{requested}:latest":
        return True
    if ":" not in requested and installed.split(":", 1)[0] == requested:
        return True
    return False


def _trim_embedding_text(text: str) -> str:
    max_chars = int(os.getenv("OLLAMA_EMBED_MAX_CHARS", str(DEFAULT_EMBED_MAX_CHARS)))
    clean = str(text)
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars]


def _extract_text(response: Dict[str, Any]) -> str:
    message = response.get("message") or {}
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return ""


def _usage_to_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    usage_keys = [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "done_reason",
    ]
    return {key: response[key] for key in usage_keys if key in response}


async def generate_answer(
    prompt: str,
    sources: Iterable[SourceSnippet],
    *,
    client: OllamaClient,
    model: str = DEFAULT_GENERATION_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> LLMResult:
    response = await client.chat(
        model=model,
        messages=build_response_input(prompt, sources),
        max_output_tokens=max_output_tokens,
    )
    text = _extract_text(response)
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return LLMResult(text=text, model=model, usage=_usage_to_dict(response))


def run_llm(query: str, context: str) -> str:
    raise RuntimeError(
        "run_llm() has been replaced by async generate_answer(); use the worker HTTP service instead"
    )
