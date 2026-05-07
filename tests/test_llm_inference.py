import pytest

from llm.inference import OllamaClient, SourceSnippet, build_response_input, generate_answer


class CapturingOllamaClient(OllamaClient):
    def __init__(self):
        self.calls = []

    async def _post(self, path, payload):
        self.calls.append((path, payload))
        return {
            "message": {"content": "final answer"},
            "prompt_eval_count": 12,
            "eval_count": 7,
        }


@pytest.mark.asyncio
async def test_generate_answer_sends_only_retrieved_snippets():
    client = CapturingOllamaClient()
    sources = [
        SourceSnippet(
            text="relevant chunk",
            source_file="book.pdf",
            page=12,
            chunk_id="chunk-1",
        )
    ]

    result = await generate_answer(
        "What is replication?",
        sources,
        client=client,
        model="llama3.2:3b-instruct-q3_K_M",
        max_output_tokens=100,
    )

    assert result.text == "final answer"
    path, payload = client.calls[0]
    assert path == "/api/chat"
    assert payload["model"] == "llama3.2:3b-instruct-q3_K_M"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 100
    assert payload["options"]["temperature"] == 0.2
    user_message = payload["messages"][1]["content"]
    assert "What is replication?" in user_message
    assert "relevant chunk" in user_message
    assert "book.pdf" in user_message
    assert "Retrieved context" in user_message
    assert result.usage["eval_count"] == 7


@pytest.mark.asyncio
async def test_ollama_embedding_response_returns_vectors(monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBED_MAX_CHARS", "20")

    class EmbeddingClient(OllamaClient):
        async def _post(self, path, payload):
            assert path == "/api/embed"
            assert payload["model"] == "all-minilm"
            assert payload["input"] == ["replication", "consensus with a lot"]
            assert payload["truncate"] is True
            return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    embeddings = await EmbeddingClient().embed_texts(
        ["replication", "consensus with a lot of extra detail"],
        model="all-minilm",
    )

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_build_response_input_without_sources_uses_prompt_only():
    messages = build_response_input("Plain question", [])

    assert messages[1]["content"] == "Plain question"
    assert "Retrieved context" not in messages[1]["content"]
