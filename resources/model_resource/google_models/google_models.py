from datetime import datetime
from numbers import Integral
from time import sleep
from typing import List, Optional

from google import genai
from google.genai import types

from resources.model_resource.model_provider import ModelProvider
from resources.model_resource.model_response import ModelResponse
from utils.logger import get_main_logger

logger = get_main_logger(__name__)

RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 60


def _valid_token_count(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        return None
    return int(value)


class GoogleModels(ModelProvider):
    def __init__(self):
        self.client = None

    def create_client(
        self, model: Optional[str] = None, system_prompt: Optional[str] = None
    ) -> genai.Client:
        return genai.Client(api_key=self._api_key())

    @staticmethod
    def _status_code(error: Exception) -> Optional[int]:
        candidates = [
            getattr(error, "status_code", None),
            getattr(getattr(error, "response", None), "status_code", None),
            getattr(error, "code", None),
        ]
        for candidate in candidates:
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue

        details = getattr(error, "details", None)
        if isinstance(details, list):
            for detail in details:
                status = getattr(detail, "status", None)
                try:
                    return int(status)
                except (TypeError, ValueError):
                    if status == "RESOURCE_EXHAUSTED":
                        return 429

        error_text = str(error)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            return 429
        return None

    def _generate_content_with_rate_limit_retry(self, **request_kwargs):
        for retry_number in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self.client.models.generate_content(**request_kwargs)
            except Exception as error:
                status_code = self._status_code(error)
                if status_code is not None:
                    error.status_code = status_code
                if status_code != 429 or retry_number == RATE_LIMIT_MAX_RETRIES:
                    raise

                logger.warning(
                    "Gemini rate limit reached. Retrying %s/%s in %s seconds.",
                    retry_number + 1,
                    RATE_LIMIT_MAX_RETRIES,
                    RATE_LIMIT_BACKOFF_SECONDS,
                )
                sleep(RATE_LIMIT_BACKOFF_SECONDS)

    def request(
        self,
        model: str,
        message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
        system_prompt: Optional[str] = None,
        thinking_level: Optional[str] = None,
    ) -> ModelResponse:
        if self.client is None:
            self.client = self.create_client()

        model_id = model.split("/", 1)[-1]
        start_time = datetime.now()
        status_code = None

        try:
            response = self._generate_content_with_rate_limit_retry(
                model=model_id,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                    max_output_tokens=max_tokens,
                    thinking_config=(
                        types.ThinkingConfig(thinking_level=thinking_level)
                        if thinking_level is not None
                        else None
                    ),
                ),
            )

            if hasattr(response, "response") and hasattr(
                response.response, "status_code"
            ):
                status_code = response.response.status_code

            end_time = datetime.now()
            response_request_duration = (end_time - start_time).total_seconds() * 1000
            usage_metadata = getattr(response, "usage_metadata", None)
            input_tokens = _valid_token_count(
                getattr(usage_metadata, "prompt_token_count", None)
            )
            if input_tokens is None:
                try:
                    token_count = self.client.models.count_tokens(
                        model=model_id, contents=message
                    )
                    input_tokens = _valid_token_count(
                        getattr(token_count, "total_tokens", None)
                    )
                except Exception as token_count_error:
                    logger.warning(
                        "Gemini input token accounting failed; recording zero: %s",
                        token_count_error,
                    )
            input_tokens = input_tokens or 0
            output_tokens = (
                _valid_token_count(
                    getattr(usage_metadata, "candidates_token_count", None)
                )
                or 0
            )
            response_text = getattr(response, "text", None)
            if not isinstance(response_text, str):
                response_text = ""

            return ModelResponse(
                content=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                time_taken_in_ms=response_request_duration,
                status_code=status_code,
            )
        except Exception as e:
            status_code = self._status_code(e)

            if status_code is not None:
                e.status_code = status_code
            raise

    def tokenize(self, model: str, message: str) -> List[int]:
        raise NotImplementedError("Tokenization is not supported for Gemini models")

    def decode(self, model: str, tokens: List[int]) -> str:
        raise NotImplementedError("Decoding tokens is not supported for Gemini models")

    def get_num_tokens(self, model: str, message: str) -> int:
        if self.client is None:
            self.client = self.create_client()
        model_id = model.split("/", 1)[-1]
        return self.client.models.count_tokens(
            model=model_id, contents=message
        ).total_tokens
