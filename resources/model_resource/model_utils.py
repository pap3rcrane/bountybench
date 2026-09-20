import http
from typing import List

import tiktoken
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from resources.model_resource.helm_models.helm_models import HelmModels
from resources.model_resource.model_provider import ModelProvider
from utils.logger import get_main_logger

logger = get_main_logger(__name__)


class ProtectedInputTruncationError(ValueError):
    """Raised rather than sending input with protected content truncated."""


def get_model_provider(use_helm: bool = False) -> ModelProvider:
    """
    Get the appropriate model provider based on the model type.
    Returns:
        ModelProvider: An instance of the appropriate model provider class.
    """
    if use_helm:
        return HelmModels()
    raise ValueError("Non-HELM tokenization uses the local tokenizer directly")


def get_local_tokenizer():
    """Return the credential-free fallback tokenizer used by non-HELM models."""
    return tiktoken.encoding_for_model("gpt-4o")


def get_num_tokens(model_input: str, model: str, use_helm: bool = False) -> int:
    """
    Returns the number of tokens for the given model input.

    Parameters:
    model_input (str): The input string to be tokenized.
    model (str): The model to be used for tokenization.
    use_helm (bool): Flag to indicate whether to use Helm provider. Default is False.

    Returns:
    int: The number of tokens in the model input.
    """
    if use_helm:
        return get_model_provider(use_helm).get_num_tokens(model, model_input)
    return len(get_local_tokenizer().encode(model_input))


@retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(
        (HTTPError, ConnectionError, Timeout, http.client.RemoteDisconnected)
    ),
)
def tokenize_input(model_input: str, model: str, use_helm: bool = False) -> List[int]:
    """
    Tokenizes the model input and returns the list of token IDs.
    Retries up to 30 times in case of certain connection errors.

    Parameters:
    model_input (str): The input string to be tokenized.
    model (str): The model to be used for tokenization.
    use_helm (bool): Flag to indicate whether to use Helm provider. Default is False.

    Returns:
    List[int]: A list of token IDs representing the tokenized input.
    """
    if use_helm:
        return get_model_provider(use_helm).tokenize(model, model_input)
    return get_local_tokenizer().encode(model_input)


@retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(
        (HTTPError, ConnectionError, Timeout, http.client.RemoteDisconnected)
    ),
)
def decode_tokenized_inputs(
    tokens: List[int], model: str, use_helm: bool = False
) -> str:
    """
    Decodes the list of token IDs back into a string.
    Retries up to 30 times in case of certain connection errors.

    Parameters:
    tokens (List[int]): A list of token IDs to be decoded.
    model (str): The model to be used for decoding.
    use_helm (bool): Flag to indicate whether to use Helm provider. Default is False.

    Returns:
    str: The decoded string from the token IDs.
    """
    if use_helm:
        return get_model_provider(use_helm).decode(model, tokens)
    return get_local_tokenizer().decode(tokens)


def truncate_input_to_max_tokens(
    max_input_tokens: int,
    model_input: str,
    model: str,
    use_helm: bool = False,
    preserve_oldest: bool = False,
    preserve_newest: bool = False,
    protected_content: str = None,
    required_prefix: str = None,
) -> str:
    if preserve_oldest and preserve_newest:
        raise ValueError("preserve_oldest and preserve_newest are mutually exclusive")
    if protected_content and protected_content not in model_input:
        raise ValueError("Protected content is not present in the model input.")
    if required_prefix and not model_input.startswith(required_prefix):
        raise ValueError("Required prefix is not at the start of the model input.")

    input_tokens = tokenize_input(model_input, model, use_helm)
    num_input_tokens = len(input_tokens)
    truncation_alert = "\n...TRUNCATED...\n"
    truncation_alert_tokens = tokenize_input(truncation_alert, model, use_helm)
    num_tokens_in_truncation_alert = len(truncation_alert_tokens)

    if num_input_tokens >= max_input_tokens - num_tokens_in_truncation_alert:
        logger.warning(
            f"Number of input tokens ({num_input_tokens}) exceeds max tokens ({max_input_tokens}). Truncating input."
        )
        tokens_to_keep = max_input_tokens - num_tokens_in_truncation_alert
        if preserve_oldest:
            truncated_tokens = input_tokens[:tokens_to_keep] + truncation_alert_tokens
            truncated_input = decode_tokenized_inputs(truncated_tokens, model, use_helm)
        elif preserve_newest:
            fixed_prefix = required_prefix or ""
            fixed_prefix_tokens = tokenize_input(fixed_prefix, model, use_helm)
            tail_tokens_to_keep = tokens_to_keep - len(fixed_prefix_tokens)
            if tail_tokens_to_keep < 0:
                raise ProtectedInputTruncationError(
                    "The required input prefix cannot fit under the "
                    f"{max_input_tokens}-token model input limit."
                )
            remaining_tokens = tokenize_input(
                model_input[len(fixed_prefix) :], model, use_helm
            )
            tail_tokens = (
                remaining_tokens[-tail_tokens_to_keep:] if tail_tokens_to_keep else []
            )
            truncated_tokens = (
                fixed_prefix_tokens + truncation_alert_tokens + tail_tokens
            )
            truncated_input = decode_tokenized_inputs(
                truncated_tokens, model, use_helm
            )
        else:
            half_tokens_to_keep = tokens_to_keep // 2
            beginning_tokens = input_tokens[:half_tokens_to_keep]
            end_tokens = input_tokens[-half_tokens_to_keep:]
            truncated_tokens = beginning_tokens + truncation_alert_tokens + end_tokens
            truncated_input = decode_tokenized_inputs(truncated_tokens, model, use_helm)

        if protected_content:
            protected_start = model_input.rfind(protected_content)
            protected_tail = model_input[protected_start:]
            required_prefix_preserved = (
                not required_prefix or truncated_input.startswith(required_prefix)
            )
            if (
                not truncated_input.endswith(protected_tail)
                or not required_prefix_preserved
            ):
                fixed_prefix = required_prefix or ""
                middle_tokens = tokenize_input(
                    model_input[len(fixed_prefix) : protected_start],
                    model,
                    use_helm,
                )
                required_input = fixed_prefix + truncation_alert + protected_tail
                required_input_tokens = tokenize_input(required_input, model, use_helm)
                middle_tokens_to_keep = max_input_tokens - len(required_input_tokens)

                if middle_tokens_to_keep >= 0:
                    protected_middle_tokens = middle_tokens[:middle_tokens_to_keep]
                    truncated_input = (
                        fixed_prefix
                        + decode_tokenized_inputs(
                            protected_middle_tokens, model, use_helm
                        )
                        + truncation_alert
                        + protected_tail
                    )

                    # Token boundaries can change when separately decoded pieces
                    # are joined. Remove middle tokens until the exact final text
                    # fits, while leaving the protected tail untouched.
                    while (
                        len(tokenize_input(truncated_input, model, use_helm))
                        > max_input_tokens
                        and protected_middle_tokens
                    ):
                        protected_middle_tokens = protected_middle_tokens[:-1]
                        truncated_input = (
                            fixed_prefix
                            + decode_tokenized_inputs(
                                protected_middle_tokens, model, use_helm
                            )
                            + truncation_alert
                            + protected_tail
                        )

                    if (
                        (fixed_prefix or protected_middle_tokens)
                        and len(tokenize_input(truncated_input, model, use_helm))
                        <= max_input_tokens
                        and truncated_input.startswith(fixed_prefix)
                        and truncated_input.endswith(protected_tail)
                    ):
                        return truncated_input

                raise ProtectedInputTruncationError(
                    "The most recent teacher message cannot fit intact with "
                    "the student prompt under the "
                    f"{max_input_tokens}-token model input limit. Rejecting "
                    "the student turn instead."
                )

        return truncated_input

    return model_input
