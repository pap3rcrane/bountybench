from datetime import datetime
from typing import List, Optional

from google import genai
from google.genai import types

from resources.model_resource.model_provider import ModelProvider
from resources.model_resource.model_response import ModelResponse


class GoogleModels(ModelProvider):
    def __init__(self):
        self.client = None

    def create_client(
        self, model: Optional[str] = None, system_prompt: Optional[str] = None
    ) -> genai.Client:
        return genai.Client(api_key=self._api_key())

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
            response = self.client.models.generate_content(
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
            input_tokens = self.client.models.count_tokens(
                model=model_id, contents=message
            ).total_tokens

            return ModelResponse(
                content=response.text,
                input_tokens=input_tokens,
                output_tokens=response.usage_metadata.candidates_token_count,
                time_taken_in_ms=response_request_duration,
                status_code=status_code,
            )
        except Exception as e:
            try:
                if hasattr(e, "status_code"):
                    status_code = e.status_code
                elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
                elif hasattr(e, "details") and isinstance(e.details, list):
                    for detail in e.details:
                        if hasattr(detail, "status") and detail.status.isdigit():
                            status_code = int(detail.status)
                            break
            except Exception:
                pass

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
