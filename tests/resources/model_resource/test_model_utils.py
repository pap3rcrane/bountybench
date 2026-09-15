from resources.model_resource import model_utils
from resources.model_resource.model_utils import (
    decode_tokenized_inputs,
    get_num_tokens,
    tokenize_input,
    truncate_input_to_max_tokens,
)

OPENROUTER_MODEL = "openrouter/deepseek/deepseek-chat-v3-0324"


class FakeTokenizer:
    def encode(self, message):
        return list(message.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


def use_fake_local_tokenizer(monkeypatch):
    monkeypatch.setattr(model_utils, "get_local_tokenizer", FakeTokenizer)


def test_openrouter_tokenization_does_not_require_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    use_fake_local_tokenizer(monkeypatch)

    message = "Credential-free local tokenization"
    tokens = tokenize_input(message, OPENROUTER_MODEL)

    assert tokens
    assert decode_tokenized_inputs(tokens, OPENROUTER_MODEL) == message
    assert get_num_tokens(message, OPENROUTER_MODEL) == len(tokens)


def test_openrouter_input_truncation_does_not_require_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    use_fake_local_tokenizer(monkeypatch)

    message = "token " * 100
    truncated = truncate_input_to_max_tokens(
        max_input_tokens=40,
        model_input=message,
        model=OPENROUTER_MODEL,
    )

    assert "...TRUNCATED..." in truncated
    assert len(tokenize_input(truncated, OPENROUTER_MODEL)) <= 40


def test_input_truncation_can_preserve_oldest_content(monkeypatch):
    use_fake_local_tokenizer(monkeypatch)

    truncated = truncate_input_to_max_tokens(
        max_input_tokens=40,
        model_input="oldest-" + ("x" * 100) + "-newest",
        model=OPENROUTER_MODEL,
        preserve_oldest=True,
    )

    assert truncated.startswith("oldest-")
    assert truncated.endswith("\n...TRUNCATED...\n")
    assert "newest" not in truncated
    assert len(tokenize_input(truncated, OPENROUTER_MODEL)) <= 40
