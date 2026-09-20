import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from agents.teacher_agent import (
    TeacherSystemPromptPlacement,
    format_teacher_trace_turn,
)
from prompts.prompts import STOP_TOKEN
from resources.model_resource.model_resource import ModelResource, ModelResourceConfig
from resources.model_resource.model_response import (
    gemini_api_response_output,
    gemini_reasoning_output,
)
from utils.logger import get_main_logger

logger = get_main_logger(__name__)
TEACHER_REQUEST_TIMEOUT_SECONDS = 900.0


@dataclass
class _ObjectiveRewriteInput:
    memory: str
    system_prompt: Optional[str] = None
    message: str = "Mock rewritten objective."
    required_memory_prefix: Optional[str] = None


@dataclass
class ObjectiveRewriteResult:
    objective: str
    teacher_trace: dict[str, Any]


def _compact_source_log(path: Path) -> tuple[str, str]:
    with path.open() as source_file:
        log = json.load(source_file)

    original_task = None
    lines = []
    turn = 0
    for phase in log.get("phase_messages", []):
        for message in phase.get("agent_messages", []):
            agent_id = message.get("agent_id")
            if agent_id == "system" and original_task is None:
                original_task = message.get("message", "")
                continue
            if agent_id != "executor_agent":
                continue
            turn += 1
            lines.append(format_teacher_trace_turn(turn))

            message_text = message.get("message", "")
            if message_text:
                lines.append(message_text)

            actions = message.get("action_messages") or []
            for action in actions:
                lines.append(f"[{action.get('resource_id', '')}]")
                lines.append(action.get("message", ""))

    if original_task is None:
        raise ValueError(f"Source log is missing the original benchmark task: {path}")
    return original_task, "\n\n".join(lines)


def _format_objective_rewrite_context(runs: list[tuple[str, str]]) -> str:
    original_task = runs[0][0].strip()
    if any(task.strip() != original_task for task, _ in runs[1:]):
        raise ValueError(
            "objective_rewrite source logs must share the same original benchmark task"
        )

    sections = [f"ORIGINAL BENCHMARK TASK:\n\n{original_task}"]
    sections.extend(
        f"## SOURCE RUN {number}: AVAILABLE TRACE (oldest to newest):\n\n"
        f"{trace.strip()}"
        for number, (_, trace) in enumerate(runs, start=1)
    )
    return "\n\n".join(sections)


def build_objective_rewrite_input(
    system_prompt_file: Optional[Path],
    system_prompt_placement: TeacherSystemPromptPlacement,
    source_logs: Iterable[Path],
) -> _ObjectiveRewriteInput:
    system_prompt_placement = TeacherSystemPromptPlacement(system_prompt_placement)
    source_logs = list(source_logs)
    if len(source_logs) != 3:
        raise ValueError("objective_rewrite requires exactly three source logs")
    if system_prompt_placement is TeacherSystemPromptPlacement.NONE:
        if system_prompt_file is not None:
            raise ValueError(
                "Teacher system prompt file must be omitted when placement is none"
            )
    elif system_prompt_file is None:
        raise ValueError(
            "Teacher system prompt file is required for prepend and system placement"
        )
    runs = [_compact_source_log(path) for path in source_logs]
    user_context = _format_objective_rewrite_context(runs)
    if system_prompt_placement is TeacherSystemPromptPlacement.PREPEND:
        assert system_prompt_file is not None
        custom_prompt = system_prompt_file.read_text().strip()
        return _ObjectiveRewriteInput(
            memory=f"{custom_prompt}\n\n{user_context}",
            required_memory_prefix=(
                f"{custom_prompt}\n\nORIGINAL BENCHMARK TASK:\n\n"
                f"{runs[0][0].strip()}\n\n"
                "## SOURCE RUN 1: AVAILABLE TRACE (oldest to newest):\n\n"
            ),
        )

    if system_prompt_placement is TeacherSystemPromptPlacement.SYSTEM:
        assert system_prompt_file is not None
        return _ObjectiveRewriteInput(
            memory=user_context,
            system_prompt=system_prompt_file.read_text().strip(),
            required_memory_prefix=(
                f"ORIGINAL BENCHMARK TASK:\n\n{runs[0][0].strip()}\n\n"
                "## SOURCE RUN 1: AVAILABLE TRACE (oldest to newest):\n\n"
            ),
        )

    return _ObjectiveRewriteInput(
        memory=user_context,
        required_memory_prefix=(
            f"ORIGINAL BENCHMARK TASK:\n\n{runs[0][0].strip()}\n\n"
            "## SOURCE RUN 1: AVAILABLE TRACE (oldest to newest):\n\n"
        ),
    )


def parse_objective_response(response: str) -> str:
    cleaned = response.strip()
    if cleaned.endswith(STOP_TOKEN):
        cleaned = cleaned[: -len(STOP_TOKEN)].rstrip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    if not cleaned:
        raise ValueError("Teacher objective must be non-empty text")

    # Continue to unwrap the old response shape when Gemini happens to use it,
    # but plain text is now the contract and never needs JSON parsing to succeed.
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return cleaned
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("objective"), str)
        and parsed["objective"].strip()
    ):
        return parsed["objective"].strip()
    return cleaned


async def rewrite_objective(
    *,
    source_logs: Iterable[Path],
    system_prompt_file: Optional[Path],
    system_prompt_placement: TeacherSystemPromptPlacement,
    model: str,
    use_mock_model: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    temperature: float,
) -> ObjectiveRewriteResult:
    source_logs = list(source_logs)
    system_prompt_placement = TeacherSystemPromptPlacement(system_prompt_placement)
    model_input = build_objective_rewrite_input(
        system_prompt_file, system_prompt_placement, source_logs
    )
    if model_input.system_prompt is not None and not model.startswith("google/"):
        raise ValueError(
            "API-level teacher system prompts require a direct google/... Gemini model"
        )

    model_resource = ModelResource(
        "teacher_model",
        ModelResourceConfig.create(
            model=model,
            use_helm=False,
            use_mock_model=use_mock_model,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout=TEACHER_REQUEST_TIMEOUT_SECONDS,
            preserve_newest_input=True,
            thinking_level="high" if model.startswith("google/") else None,
        ),
    )
    try:
        action = await asyncio.to_thread(model_resource.run, model_input)
    finally:
        model_resource.stop()

    logger.info(
        "\n%s\nTEACHER RESPONSE (objective_rewrite)\n%s\n%s",
        "=" * 80,
        action.message,
        "=" * 80,
    )
    objective = parse_objective_response(action.message)
    action_metadata = action.additional_metadata or {}
    trace_model = action_metadata.get("model", model)
    reasoning_output = action_metadata.get("reasoning_output")
    if reasoning_output is None and trace_model.startswith("google/"):
        reasoning_output = gemini_reasoning_output([])
    gemini_api_response = action_metadata.get("gemini_api_response")
    if gemini_api_response is None and trace_model.startswith("google/"):
        gemini_api_response = gemini_api_response_output(
            response_type="google.genai.types.GenerateContentResponse",
            data=None,
            unavailable_reason="Gemini provider returned no serialized API response",
        )
    teacher_trace = {
        "teacher_mode": "objective_rewrite",
        "model": trace_model,
        "system_prompt_placement": system_prompt_placement.value,
        "system_prompt": action_metadata.get(
            "system_prompt", model_input.system_prompt
        ),
        "input": action_metadata.get("input", model_input.memory),
        "raw_response": action_metadata.get("raw_output", action.message),
        "temperature": action_metadata.get("temperature", temperature),
        "thinking_level": action_metadata.get(
            "thinking_level", "high" if model.startswith("google/") else None
        ),
        "source_logs": [str(path) for path in source_logs],
    }
    if reasoning_output is not None:
        teacher_trace["reasoning_output"] = reasoning_output
    if gemini_api_response is not None:
        teacher_trace["gemini_api_response"] = gemini_api_response
    return ObjectiveRewriteResult(
        objective=objective,
        teacher_trace=teacher_trace,
    )


def write_objective(
    *,
    objective: str,
    output_dir: Path,
    task_dir: Path,
    bounty_number: str,
    workflow_type: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f%z")
    output_path = output_dir / (
        f"{task_dir.name}_bounty_{bounty_number}_{workflow_type}_{timestamp}_objective.json"
    )
    output_path.write_text(json.dumps({"objective": objective}, indent=2) + "\n")
    return output_path


def write_teacher_trace(*, teacher_trace: dict[str, Any], objective_path: Path) -> Path:
    suffix = "_objective.json"
    trace_name = (
        objective_path.name[: -len(suffix)] + "_teacher_trace.json"
        if objective_path.name.endswith(suffix)
        else objective_path.stem + "_teacher_trace.json"
    )
    trace_path = objective_path.with_name(trace_name)
    trace_path.write_text(json.dumps(teacher_trace, indent=2) + "\n")
    return trace_path
