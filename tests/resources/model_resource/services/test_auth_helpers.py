from unittest.mock import Mock, patch

from resources.model_resource.services.auth_helpers import _auth_google_api_key


def test_google_quota_response_still_authenticates_key():
    response = Mock(status_code=429, text="quota exceeded")

    with patch(
        "resources.model_resource.services.auth_helpers.requests.get",
        return_value=response,
    ):
        valid, message = _auth_google_api_key(
            "test-key", "google/gemini-3.6-flash", verify_model=True
        )

    assert valid is True
    assert message == ""
