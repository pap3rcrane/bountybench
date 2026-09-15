import unittest
from unittest.mock import MagicMock, patch

from resources.model_resource.model_mapping import get_model_info
from resources.model_resource.model_resource import ModelResource
from resources.model_resource.openrouter_models.openrouter_models import (
    OPENROUTER_BASE_URL,
    OpenRouterModels,
)
from resources.model_resource.services.auth_helpers import _auth_openrouter_api_key
from resources.model_resource.services.service_providers import ServiceProvider


class TestOpenRouterModels(unittest.TestCase):
    @patch("resources.model_resource.openrouter_models.openrouter_models.OpenAI")
    @patch.object(OpenRouterModels, "_api_key", return_value="test-openrouter-key")
    def test_request_uses_openrouter_slug(self, mock_api_key, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.response = None
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenRouterModels()
        response = provider.request(
            model="openrouter/deepseek/deepseek-chat-v3-0324",
            message="Test message",
            temperature=0.5,
            max_tokens=100,
            stop_sequences=["STOP"],
        )

        mock_openai.assert_called_once_with(
            api_key="test-openrouter-key", base_url=OPENROUTER_BASE_URL
        )
        mock_client.chat.completions.create.assert_called_once_with(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.5,
            max_tokens=100,
            stop=["STOP"],
        )
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.input_tokens, 10)
        self.assertEqual(response.output_tokens, 20)

    @patch("resources.model_resource.services.auth_helpers.requests.get")
    def test_auth_verifies_registered_model(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [{"id": "deepseek/deepseek-chat-v3-0324"}]
        }

        ok, message = _auth_openrouter_api_key(
            "test-openrouter-key",
            "openrouter/deepseek/deepseek-chat-v3-0324",
            verify_model=True,
        )

        self.assertTrue(ok)
        self.assertEqual(message, "")
        mock_get.assert_called_once_with(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": "Bearer test-openrouter-key"},
        )

    def test_model_is_registered_as_openrouter(self):
        model_info = get_model_info(
            "openrouter/deepseek/deepseek-chat-v3-0324", helm=False
        )

        self.assertEqual(model_info.provider, ServiceProvider.OPENROUTER)
        self.assertEqual(
            model_info.model_name,
            "openrouter/deepseek/deepseek-chat-v3-0324",
        )

    @patch("resources.model_resource.openrouter_models.OpenRouterModels")
    def test_model_resource_selects_openrouter_provider(self, mock_provider_class):
        resource = object.__new__(ModelResource)
        resource.model = "openrouter/deepseek/deepseek-chat-v3-0324"
        resource.helm = False
        resource.use_mock_model = False

        provider = resource.get_model_provider()

        self.assertIs(provider, mock_provider_class.return_value)
        mock_provider_class.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
