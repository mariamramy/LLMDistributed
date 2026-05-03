from types import SimpleNamespace

import pytest

from llm.inference import SourceSnippet, build_response_input, generate_answer


class FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="final answer", usage={"output_tokens": 7})


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


@pytest.mark.asyncio
async def test_generate_answer_sends_only_retrieved_snippets():
    client = FakeOpenAIClient()
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
        model="gpt-5.4-nano",
        max_output_tokens=100,
    )

    assert result.text == "final answer"
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.4-nano"
    user_message = call["input"][1]["content"]
    assert "What is replication?" in user_message
    assert "relevant chunk" in user_message
    assert "book.pdf" in user_message
    assert "Retrieved context" in user_message


def test_build_response_input_without_sources_uses_prompt_only():
    messages = build_response_input("Plain question", [])

    assert messages[1]["content"] == "Plain question"
    assert "Retrieved context" not in messages[1]["content"]
