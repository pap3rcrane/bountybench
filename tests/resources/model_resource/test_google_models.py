from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from google import genai
from google.genai import types

from resources.model_resource.google_models.google_models import GoogleModels
from resources.model_resource.model_mapping import get_model_info
from resources.model_resource.model_resource import ModelResource, ModelResourceConfig
from resources.model_resource.model_response import ModelResponse
from resources.model_resource.services.service_providers import ServiceProvider


def test_gemini_36_flash_is_registered_as_a_direct_google_model():
    model = get_model_info("google/gemini-3.6-flash", helm=False)

    assert model.model_name == "gemini-3.6-flash"
    assert model.provider is ServiceProvider.GOOGLE


def test_gemini_system_prompt_is_passed_as_system_instruction():
    response = MagicMock()
    response.text = "Teacher response"
    response.usage_metadata.candidates_token_count = 4
    client = MagicMock()
    client.models.generate_content.return_value = response
    client.models.count_tokens.return_value.total_tokens = 3

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client) as client_class,
    ):
        provider = GoogleModels()
        provider.request(
            model="google/gemini-2.5-flash",
            message="TRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
            system_prompt="CUSTOM TEACHER SYSTEM PROMPT",
        )

    client_class.assert_called_once_with(api_key="test-key")
    request = client.models.generate_content.call_args.kwargs
    assert request["model"] == "gemini-2.5-flash"
    assert request["contents"] == ("TRACE AND INSTRUCTIONS")
    assert request["config"].system_instruction == "CUSTOM TEACHER SYSTEM PROMPT"


def test_gemini_prepend_transport_does_not_set_a_system_instruction():
    response = MagicMock()
    response.text = "Teacher response"
    response.usage_metadata.candidates_token_count = 4
    client = MagicMock()
    client.models.generate_content.return_value = response
    client.models.count_tokens.return_value.total_tokens = 3

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
    ):
        provider = GoogleModels()
        provider.request(
            model="google/gemini-2.5-flash",
            message="CUSTOM PROMPT\n\nTRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
        )

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.system_instruction is None


def test_gemini_normalizes_missing_text_and_usage_counts():
    response = MagicMock()
    response.text = None
    response.usage_metadata = None
    client = MagicMock()
    client.models.generate_content.return_value = response
    client.models.count_tokens.return_value.total_tokens = None

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
    ):
        result = GoogleModels().request(
            model="google/gemini-3.6-flash",
            message="TRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
        )

    assert result.content == ""
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_gemini_keeps_response_when_fallback_token_counting_fails():
    response = MagicMock()
    response.text = "Teacher response"
    response.usage_metadata = None
    client = MagicMock()
    client.models.generate_content.return_value = response
    client.models.count_tokens.side_effect = RuntimeError("counting unavailable")

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
    ):
        result = GoogleModels().request(
            model="google/gemini-3.6-flash",
            message="TRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
        )

    assert result.content == "Teacher response"
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_gemini_high_thinking_is_passed_in_generation_config():
    response = MagicMock()
    response.text = "Teacher response"
    response.usage_metadata.candidates_token_count = 4
    client = MagicMock()
    client.models.generate_content.return_value = response
    client.models.count_tokens.return_value.total_tokens = 3

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
    ):
        GoogleModels().request(
            model="google/gemini-3.6-flash",
            message="TRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
            thinking_level="high",
        )

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.thinking_config.thinking_level is types.ThinkingLevel.HIGH


def test_gemini_retries_rate_limits_three_times_with_fixed_backoff():
    rate_limit_error = Exception("429 RESOURCE_EXHAUSTED")
    response = MagicMock()
    response.text = "Teacher response"
    response.usage_metadata.candidates_token_count = 4
    client = MagicMock()
    client.models.generate_content.side_effect = [
        rate_limit_error,
        rate_limit_error,
        rate_limit_error,
        response,
    ]
    client.models.count_tokens.return_value.total_tokens = 3

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
        patch(
            "resources.model_resource.google_models.google_models.sleep"
        ) as mock_sleep,
    ):
        result = GoogleModels().request(
            model="google/gemini-3.6-flash",
            message="TRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
        )

    assert result.content == "Teacher response"
    assert client.models.generate_content.call_count == 4
    assert mock_sleep.call_args_list == [call(60), call(60), call(60)]


def test_gemini_raises_after_three_rate_limit_retries():
    rate_limit_error = Exception("429 RESOURCE_EXHAUSTED")
    client = MagicMock()
    client.models.generate_content.side_effect = rate_limit_error

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
        patch(
            "resources.model_resource.google_models.google_models.sleep"
        ) as mock_sleep,
    ):
        with pytest.raises(Exception, match="RESOURCE_EXHAUSTED") as error:
            GoogleModels().request(
                model="google/gemini-3.6-flash",
                message="TRACE AND INSTRUCTIONS",
                temperature=0.0,
                max_tokens=100,
                stop_sequences=[],
            )

    assert error.value.status_code == 429
    assert client.models.generate_content.call_count == 4
    assert mock_sleep.call_args_list == [call(60), call(60), call(60)]


def test_gemini_retries_service_unavailable_three_times_with_fixed_backoff():
    unavailable_error = Exception("503 UNAVAILABLE")
    response = MagicMock()
    response.text = "Teacher response"
    response.usage_metadata.candidates_token_count = 4
    client = MagicMock()
    client.models.generate_content.side_effect = [
        unavailable_error,
        unavailable_error,
        unavailable_error,
        response,
    ]
    client.models.count_tokens.return_value.total_tokens = 3

    with (
        patch.object(GoogleModels, "_api_key", return_value="test-key"),
        patch.object(genai, "Client", return_value=client),
        patch(
            "resources.model_resource.google_models.google_models.sleep"
        ) as mock_sleep,
    ):
        result = GoogleModels().request(
            model="google/gemini-3.6-flash",
            message="TRACE AND INSTRUCTIONS",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=[],
        )

    assert result.content == "Teacher response"
    assert client.models.generate_content.call_count == 4
    assert mock_sleep.call_args_list == [call(60), call(60), call(60)]


def test_model_resource_forwards_separate_gemini_system_prompt():
    provider = MagicMock()
    provider.make_request.return_value = ModelResponse(
        content="Teacher response",
        input_tokens=3,
        output_tokens=4,
        time_taken_in_ms=5,
    )

    with (
        patch("resources.model_resource.model_resource.verify_and_auth_api_key"),
        patch.object(ModelResource, "get_model_provider", return_value=provider),
        patch(
            "resources.model_resource.model_resource.truncate_input_to_max_tokens",
            return_value="USER MESSAGE",
        ),
    ):
        resource = ModelResource(
            "teacher_model",
            ModelResourceConfig(model="google/gemini-2.5-flash"),
        )
        action = resource.run(
            SimpleNamespace(
                memory="USER MESSAGE",
                system_prompt="CUSTOM TEACHER SYSTEM PROMPT",
            )
        )

    assert provider.make_request.call_args.kwargs["system_prompt"] == (
        "CUSTOM TEACHER SYSTEM PROMPT"
    )
    assert action.additional_metadata["system_prompt"] == (
        "CUSTOM TEACHER SYSTEM PROMPT"
    )


def test_direct_gemini_reserves_input_headroom_for_tokenizer_differences():
    provider = MagicMock()
    provider.make_request.return_value = ModelResponse(
        content="Teacher response",
        input_tokens=3,
        output_tokens=4,
        time_taken_in_ms=5,
    )

    with (
        patch("resources.model_resource.model_resource.verify_and_auth_api_key"),
        patch.object(ModelResource, "get_model_provider", return_value=provider),
        patch(
            "resources.model_resource.model_resource.truncate_input_to_max_tokens",
            return_value="USER MESSAGE",
        ) as truncate,
    ):
        resource = ModelResource(
            "teacher_model",
            ModelResourceConfig(
                model="google/gemini-3.6-flash", max_input_tokens=1_048_576
            ),
        )
        resource.run(SimpleNamespace(memory="USER MESSAGE"))

    assert truncate.call_args.kwargs["max_input_tokens"] == 943_718


def test_model_resource_forwards_high_thinking_level():
    provider = MagicMock()
    provider.make_request.return_value = ModelResponse(
        content="Teacher response",
        input_tokens=3,
        output_tokens=4,
        time_taken_in_ms=5,
    )

    with (
        patch("resources.model_resource.model_resource.verify_and_auth_api_key"),
        patch.object(ModelResource, "get_model_provider", return_value=provider),
        patch(
            "resources.model_resource.model_resource.truncate_input_to_max_tokens",
            return_value="USER MESSAGE",
        ),
    ):
        resource = ModelResource(
            "teacher_model",
            ModelResourceConfig(model="google/gemini-3.6-flash", thinking_level="high"),
        )
        action = resource.run(SimpleNamespace(memory="USER MESSAGE"))

    assert provider.make_request.call_args.kwargs["thinking_level"] == "high"
    assert action.additional_metadata["thinking_level"] == "high"
