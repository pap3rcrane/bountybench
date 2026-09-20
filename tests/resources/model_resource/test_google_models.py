import json
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


def test_gemini_thought_summaries_are_requested_and_separated_from_answer():
    response = SimpleNamespace(
        text="SDK fallback text",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="First inspect the trace.", thought=True),
                        SimpleNamespace(
                            text="Then compare the evidence.", thought=True
                        ),
                        SimpleNamespace(text="Teacher response", thought=False),
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=3,
            candidates_token_count=4,
        ),
    )
    client = MagicMock()
    client.models.generate_content.return_value = response

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
            thinking_level="high",
            include_thoughts=True,
        )

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.thinking_config.include_thoughts is True
    assert result.content == "Teacher response"
    assert result.reasoning_output == {
        "type": "gemini_thought_summary",
        "available": True,
        "text": "First inspect the trace.\n\nThen compare the evidence.",
        "parts": ["First inspect the trace.", "Then compare the evidence."],
    }


def test_gemini_records_complete_json_serializable_sdk_response():
    response = types.GenerateContentResponse(
        sdk_http_response=types.HttpResponse(
            headers={"x-request-id": "request-123"},
            body='{"responseId":"response-123"}',
        ),
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text="Inspect the evidence.",
                            thought=True,
                            thought_signature=b"signature",
                        ),
                        types.Part(text="Teacher response"),
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
                index=0,
                safety_ratings=[
                    types.SafetyRating(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        probability=types.HarmProbability.NEGLIGIBLE,
                    )
                ],
            )
        ],
        model_version="gemini-test-version",
        response_id="response-123",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3,
            candidates_token_count=4,
            thoughts_token_count=2,
            total_token_count=9,
        ),
    )
    client = MagicMock()
    client.models.generate_content.return_value = response

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
            thinking_level="high",
            include_thoughts=True,
        )

    recorded = result.gemini_api_response
    assert recorded["type"] == "google.genai.types.GenerateContentResponse"
    assert recorded["serialization"] == (
        "pydantic.model_dump(mode=json, by_alias=false, exclude_none=false)"
    )
    assert recorded["available"] is True
    assert recorded["data"]["response_id"] == "response-123"
    assert recorded["data"]["model_version"] == "gemini-test-version"
    assert recorded["data"]["sdk_http_response"] == {
        "headers": {"x-request-id": "request-123"},
        "body": '{"responseId":"response-123"}',
    }
    candidate = recorded["data"]["candidates"][0]
    assert candidate["finish_reason"] == "STOP"
    assert candidate["content"]["parts"][0]["thought"] is True
    assert candidate["content"]["parts"][0]["thought_signature"] == ("c2lnbmF0dXJl")
    assert candidate["safety_ratings"][0]["probability"] == "NEGLIGIBLE"
    assert recorded["data"]["usage_metadata"] == {
        "cache_tokens_details": None,
        "cached_content_token_count": None,
        "candidates_token_count": 4,
        "candidates_tokens_details": None,
        "prompt_token_count": 3,
        "prompt_tokens_details": None,
        "thoughts_token_count": 2,
        "tool_use_prompt_token_count": None,
        "tool_use_prompt_tokens_details": None,
        "total_token_count": 9,
        "traffic_type": None,
    }
    assert recorded["data"]["prompt_feedback"] is None
    json.dumps(recorded)


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
    assert not any("token" in key for key in action.additional_metadata)


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
    api_response = {
        "type": "google.genai.types.GenerateContentResponse",
        "serialization": (
            "pydantic.model_dump(mode=json, by_alias=false, exclude_none=false)"
        ),
        "available": True,
        "data": {"response_id": "response-123"},
    }
    provider = MagicMock()
    provider.make_request.return_value = ModelResponse(
        content="Teacher response",
        input_tokens=3,
        output_tokens=4,
        time_taken_in_ms=5,
        gemini_api_response=api_response,
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
    assert provider.make_request.call_args.kwargs["include_thoughts"] is True
    assert action.additional_metadata["thinking_level"] == "high"
    assert action.additional_metadata["reasoning_output"] == {
        "type": "gemini_thought_summary",
        "available": False,
        "text": "",
        "parts": [],
    }
    assert action.additional_metadata["gemini_api_response"] == api_response


def test_non_teacher_gemini_does_not_request_or_record_thought_summaries():
    provider = MagicMock()
    provider.make_request.return_value = ModelResponse(
        content="Student response",
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
            "model",
            ModelResourceConfig(model="google/gemini-3.6-flash"),
        )
        action = resource.run(SimpleNamespace(memory="USER MESSAGE"))

    assert provider.make_request.call_args.kwargs["include_thoughts"] is None
    assert "reasoning_output" not in action.additional_metadata
    assert "gemini_api_response" not in action.additional_metadata


def test_mock_gemini_teacher_records_unavailable_reasoning_output():
    resource = ModelResource(
        "teacher_model",
        ModelResourceConfig(
            model="google/gemini-3.6-flash",
            use_mock_model=True,
        ),
    )

    action = resource.run(
        SimpleNamespace(memory="USER MESSAGE", message="Mock teacher response")
    )

    assert action.additional_metadata["reasoning_output"] == {
        "type": "gemini_thought_summary",
        "available": False,
        "text": "",
        "parts": [],
    }
    assert action.additional_metadata["gemini_api_response"] == {
        "type": "google.genai.types.GenerateContentResponse",
        "serialization": (
            "pydantic.model_dump(mode=json, by_alias=false, exclude_none=false)"
        ),
        "available": False,
        "data": None,
        "unavailable_reason": "mock model; no Gemini API call was made",
    }
