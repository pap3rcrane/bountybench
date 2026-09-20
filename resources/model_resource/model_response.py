from dataclasses import dataclass
from typing import Any, Iterable, Optional

HALLUCINATION_STRINGS = [
    "----------Message from assistant----------",
    "----------Message from system----------",
    "----------Message from assistant",
    "----------</assistant",
    "----------Message from agent",
]

GEMINI_THOUGHT_SUMMARY_TYPE = "gemini_thought_summary"
GEMINI_API_RESPONSE_SERIALIZATION = (
    "pydantic.model_dump(mode=json, by_alias=false, exclude_none=false)"
)


def gemini_reasoning_output(parts: Iterable[str]) -> dict[str, Any]:
    """Return the stable trace representation for Gemini thought summaries."""
    recorded_parts = [part for part in parts if isinstance(part, str) and part]
    return {
        "type": GEMINI_THOUGHT_SUMMARY_TYPE,
        "available": bool(recorded_parts),
        "text": "\n\n".join(recorded_parts),
        "parts": recorded_parts,
    }


def gemini_api_response_output(
    response_type: str,
    data: Optional[dict[str, Any]],
    unavailable_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Return the stable trace envelope for a complete Gemini SDK response."""
    output = {
        "type": response_type,
        "serialization": GEMINI_API_RESPONSE_SERIALIZATION,
        "available": data is not None,
        "data": data,
    }
    if unavailable_reason is not None:
        output["unavailable_reason"] = unavailable_reason
    return output


@dataclass(frozen=True)
class ModelResponse:
    """
    Dataclass of model response
    """

    content: str
    input_tokens: int
    output_tokens: int
    time_taken_in_ms: float
    status_code: Optional[int] = None
    reasoning_output: Optional[dict[str, Any]] = None
    gemini_api_response: Optional[dict[str, Any]] = None

    def remove_hallucinations(self):
        response = self.content
        response = response.replace("----------Message from agent----------", "")
        for hallucination in HALLUCINATION_STRINGS:
            hallucination_index = response.find(hallucination)
            if hallucination_index != -1:
                response = response[:hallucination_index]
        return response.strip()

    @staticmethod
    def from_dict(d: dict) -> "ModelResponse":
        return ModelResponse(
            d["content"],
            d["input_tokens"],
            d["output_tokens"],
            d["time_taken_in_ms"],
            d.get("status_code"),
            d.get("reasoning_output"),
            d.get("gemini_api_response"),
        )

    def to_dict(self):
        result = {
            "content": self.content,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "time_taken_in_ms": self.time_taken_in_ms,
        }
        if self.status_code is not None:
            result["status_code"] = self.status_code
        if self.reasoning_output is not None:
            result["reasoning_output"] = self.reasoning_output
        if self.gemini_api_response is not None:
            result["gemini_api_response"] = self.gemini_api_response
        return result
