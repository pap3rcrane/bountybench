import pytest

from prompts.teacher_response_context import format_teacher_output_for_student
from resources.model_resource import model_utils
from resources.model_resource.model_utils import (
    ProtectedInputTruncationError,
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


def test_input_truncation_can_preserve_newest_content(monkeypatch):
    use_fake_local_tokenizer(monkeypatch)

    truncated = truncate_input_to_max_tokens(
        max_input_tokens=48,
        model_input="trace-header\noldest-" + ("x" * 100) + "-newest",
        model=OPENROUTER_MODEL,
        preserve_newest=True,
        required_prefix="trace-header\n",
    )

    assert truncated.startswith("trace-header\n\n...TRUNCATED...\n")
    assert truncated.endswith("-newest")
    assert "oldest" not in truncated
    assert len(tokenize_input(truncated, OPENROUTER_MODEL)) <= 48


def test_input_truncation_keeps_complete_protected_teacher_tail(monkeypatch):
    use_fake_local_tokenizer(monkeypatch)
    teacher = "[teacher_agent] " + format_teacher_output_for_student(
        "observe",
        "prompts/system_prompts/single_error_correction_one_alternative.txt",
        "complete feedback",
    )
    retry_reminder = '\n\nInclude "Command:" in your response.'
    message = "student prompt\n" + ("history " * 30) + teacher + retry_reminder

    truncated = truncate_input_to_max_tokens(
        max_input_tokens=400,
        model_input=message,
        model=OPENROUTER_MODEL,
        protected_content=teacher,
        required_prefix="student prompt\n",
    )

    assert "...TRUNCATED..." in truncated
    assert truncated.startswith("student prompt\n")
    assert truncated.endswith(teacher + retry_reminder)
    assert len(tokenize_input(truncated, OPENROUTER_MODEL)) <= 400


def test_input_truncation_rejects_instead_of_shortening_teacher(monkeypatch):
    use_fake_local_tokenizer(monkeypatch)
    teacher = "[teacher_agent] " + format_teacher_output_for_student(
        "observe",
        "prompts/system_prompts/single_error_correction_one_alternative.txt",
        "feedback" * 20,
    )
    message = "student prompt\n" + ("history " * 30) + teacher

    with pytest.raises(
        ProtectedInputTruncationError,
        match="most recent teacher message cannot fit intact",
    ):
        truncate_input_to_max_tokens(
            max_input_tokens=140,
            model_input=message,
            model=OPENROUTER_MODEL,
            protected_content=teacher,
            required_prefix="student prompt\n",
        )
