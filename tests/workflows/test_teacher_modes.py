import asyncio
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents.teacher_agent import TeacherSystemPromptPlacement
from messages.action_messages.action_message import ActionMessage
from prompts.prompts import (
    DETECT_DESCRIPTION,
    DETECT_PROMPT,
    EXPLOIT_AND_PATCH_PROMPT,
    EXPLOIT_DESCRIPTION,
    replace_objective_description,
)
from workflows.runner import WorkflowRunner
from workflows.teacher_modes import (
    build_objective_rewrite_input,
    parse_objective_response,
    rewrite_objective,
    write_objective,
)

OBJECTIVE_RESPONSE_FORMAT = (
    "Return only the rewritten objective as plain text, with no JSON wrapper or "
    "surrounding commentary."
)


def _source_log(path, command):
    path.write_text(
        json.dumps(
            {
                "workflow_metadata": {"workflow_name": "DetectWorkflow"},
                "phase_messages": [
                    {
                        "phase_id": "DetectPhase",
                        "phase_summary": "no_submission/failure",
                        "success": False,
                        "agent_messages": [
                            {
                                "agent_id": "system",
                                "message": f"Original task for {command}",
                            },
                            {
                                "agent_id": "executor_agent",
                                "message": "",
                                "action_messages": [
                                    {"resource_id": "model", "message": command}
                                ],
                            },
                            {
                                "agent_id": "teacher_agent",
                                "message": "must not be reused",
                            },
                            {
                                "agent_id": "detect_agent",
                                "message": "evaluator must not be reused",
                            },
                        ],
                    }
                ],
            }
        )
    )


def test_objective_replaces_only_workflow_description():
    detect = replace_objective_description(
        DETECT_PROMPT, "DetectWorkflow", "NEW OBJECTIVE"
    )
    combined = replace_objective_description(
        EXPLOIT_AND_PATCH_PROMPT, "ExploitPatchWorkflow", "COMBINED OBJECTIVE"
    )

    assert DETECT_DESCRIPTION not in detect
    assert detect.startswith("NEW OBJECTIVE")
    assert "Always respond in this format exactly" in detect
    assert EXPLOIT_DESCRIPTION not in combined
    assert combined.startswith("COMBINED OBJECTIVE")
    assert "exploit and patch" in combined


def test_objective_response_accepts_plain_text_and_legacy_json():
    assert parse_objective_response("Improve this.\n<END>") == "Improve this."
    assert parse_objective_response('{"objective": "Improve this."}\n<END>') == (
        "Improve this."
    )
    assert parse_objective_response("```text\nImprove this.\n```") == "Improve this."
    assert parse_objective_response("not valid JSON") == "not valid JSON"
    with pytest.raises(ValueError, match="non-empty"):
        parse_objective_response("  \n<END>")


def test_objective_input_uses_three_logs_and_removes_old_teacher_turns(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(f"Rewrite weak objectives.\n\n{OBJECTIVE_RESPONSE_FORMAT}")
    logs = []
    for number in range(3):
        path = tmp_path / f"run-{number}.json"
        _source_log(path, f"Command: command-{number}")
        logs.append(path)

    model_input = build_objective_rewrite_input(
        prompt, TeacherSystemPromptPlacement.PREPEND, logs
    )
    contents = model_input.memory

    assert model_input.system_prompt is None
    assert contents.count("ORIGINAL BENCHMARK TASK:") == 3
    assert contents.count("AVAILABLE TRACE (oldest to newest):") == 3
    assert contents.count("TURN 1 agent=executor_agent") == 3
    assert "TURN 2 " not in contents
    assert "command-0" in contents
    assert "command-1" in contents
    assert "command-2" in contents
    assert "must not be reused" not in contents
    assert "evaluator must not be reused" not in contents
    assert OBJECTIVE_RESPONSE_FORMAT in contents


def test_objective_system_placement_keeps_system_prompt_out_of_user_input(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(f"ACTUAL SYSTEM PROMPT\n\n{OBJECTIVE_RESPONSE_FORMAT}")
    logs = []
    for number in range(3):
        path = tmp_path / f"run-{number}.json"
        _source_log(path, f"Command: command-{number}")
        logs.append(path)

    model_input = build_objective_rewrite_input(
        prompt, TeacherSystemPromptPlacement.SYSTEM, logs
    )

    assert "ACTUAL SYSTEM PROMPT" not in model_input.memory
    assert "ORIGINAL BENCHMARK TASK:" not in model_input.memory
    assert model_input.memory.count("AVAILABLE TRACE (oldest to newest):") == 3
    assert model_input.system_prompt.startswith("ACTUAL SYSTEM PROMPT\n\n")
    assert model_input.system_prompt.count("ORIGINAL BENCHMARK TASK:") == 3


def test_objective_none_placement_needs_no_system_prompt_file(tmp_path):
    logs = []
    for number in range(3):
        path = tmp_path / f"run-{number}.json"
        _source_log(path, f"Command: command-{number}")
        logs.append(path)

    model_input = build_objective_rewrite_input(
        None, TeacherSystemPromptPlacement.NONE, logs
    )

    assert model_input.system_prompt is None
    assert model_input.memory.count("ORIGINAL BENCHMARK TASK:") == 3
    assert model_input.memory.count("AVAILABLE TRACE (oldest to newest):") == 3


def test_objective_rewrite_passes_separate_system_prompt_to_model(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(f"ACTUAL SYSTEM PROMPT\n\n{OBJECTIVE_RESPONSE_FORMAT}")
    logs = []
    for number in range(3):
        path = tmp_path / f"run-{number}.json"
        _source_log(path, f"Command: command-{number}")
        logs.append(path)

    with patch("workflows.teacher_modes.ModelResource") as model_resource:
        model_resource.return_value.run.return_value = ActionMessage(
            resource_id="teacher_model",
            message="Better objective.",
            additional_metadata={
                "gemini_api_response": {
                    "type": "google.genai.types.GenerateContentResponse",
                    "serialization": (
                        "pydantic.model_dump(mode=json, by_alias=false, "
                        "exclude_none=false)"
                    ),
                    "available": True,
                    "data": {"response_id": "response-123"},
                },
                "reasoning_output": {
                    "type": "gemini_thought_summary",
                    "available": True,
                    "text": "The original objective is underspecified.",
                    "parts": ["The original objective is underspecified."],
                },
            },
        )
        result = asyncio.run(
            rewrite_objective(
                source_logs=logs,
                system_prompt_file=prompt,
                system_prompt_placement=TeacherSystemPromptPlacement.SYSTEM,
                model="google/gemini-2.5-flash",
                use_mock_model=True,
                max_input_tokens=1000,
                max_output_tokens=100,
                temperature=0.0,
            )
        )

    model_input = model_resource.return_value.run.call_args.args[0]
    model_config = model_resource.call_args.args[1]
    assert result.objective == "Better objective."
    assert result.teacher_trace["input"] == model_input.memory
    assert result.teacher_trace["system_prompt"] == model_input.system_prompt
    assert result.teacher_trace["raw_response"] == "Better objective."
    assert result.teacher_trace["reasoning_output"] == {
        "type": "gemini_thought_summary",
        "available": True,
        "text": "The original objective is underspecified.",
        "parts": ["The original objective is underspecified."],
    }
    assert result.teacher_trace["gemini_api_response"] == {
        "type": "google.genai.types.GenerateContentResponse",
        "serialization": (
            "pydantic.model_dump(mode=json, by_alias=false, exclude_none=false)"
        ),
        "available": True,
        "data": {"response_id": "response-123"},
    }
    assert "response" not in result.teacher_trace
    assert "objective" not in result.teacher_trace
    assert model_config.preserve_oldest_input is True
    assert model_config.thinking_level == "high"
    assert model_config.timeout == 900.0
    assert model_input.system_prompt.startswith("ACTUAL SYSTEM PROMPT\n\n")
    assert model_input.system_prompt.count("ORIGINAL BENCHMARK TASK:") == 3
    assert "ACTUAL SYSTEM PROMPT" not in model_input.memory


@pytest.mark.parametrize(
    "prompt_name",
    [
        "increased_difficulty_task",
        "weakness_targeting_task",
        "strategy_limiter_task",
        "justification_gap_task",
        "oversight_task",
    ],
)
def test_objective_task_prompts_end_with_plain_text_response_format(prompt_name):
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "system_prompts"
        / f"{prompt_name}.txt"
    )
    assert prompt_path.read_text().strip().endswith(OBJECTIVE_RESPONSE_FORMAT)


def test_written_objective_contains_only_objective(tmp_path):
    path = write_objective(
        objective="New objective",
        output_dir=tmp_path,
        task_dir=tmp_path / "lunary",
        bounty_number="0",
        workflow_type="detect_workflow",
    )
    assert json.loads(path.read_text()) == {"objective": "New objective"}


def test_teacher_mode_and_placement_are_required_together():
    runner = WorkflowRunner()
    runner.args = Namespace(
        teacher_mode="observe",
        teacher_system_prompt_file=None,
        teacher_system_prompt_placement=None,
        teacher_model="google/gemini-2.5-flash",
        source_logs=None,
        generate_source_runs=False,
    )
    with pytest.raises(SystemExit):
        runner._validate_teacher_arguments()

    runner.args.teacher_system_prompt_file = "teacher.txt"
    with pytest.raises(SystemExit):
        runner._validate_teacher_arguments()

    runner.args.teacher_system_prompt_placement = "none"
    with pytest.raises(SystemExit):
        runner._validate_teacher_arguments()

    runner.args.teacher_system_prompt_file = None
    runner._validate_teacher_arguments()


def test_system_placement_requires_direct_gemini_teacher():
    runner = WorkflowRunner()
    runner.args = Namespace(
        teacher_mode="observe",
        teacher_system_prompt_file="teacher.txt",
        teacher_system_prompt_placement="system",
        teacher_model="openrouter/google/gemini-2.5-flash",
        source_logs=None,
        generate_source_runs=False,
    )

    with pytest.raises(SystemExit):
        runner._validate_teacher_arguments()

    runner.args.teacher_model = "google/gemini-2.5-flash"
    runner._validate_teacher_arguments()


def test_objective_rewrite_requires_one_source_strategy():
    runner = WorkflowRunner()
    runner.args = Namespace(
        teacher_mode="objective_rewrite",
        teacher_system_prompt_file="teacher.txt",
        teacher_system_prompt_placement="prepend",
        teacher_model="teacher",
        source_logs=None,
        generate_source_runs=False,
    )
    with pytest.raises(SystemExit):
        runner._validate_teacher_arguments()

    runner.args.generate_source_runs = True
    runner._validate_teacher_arguments()


def test_student_kwargs_remove_all_teacher_behavior():
    assert WorkflowRunner._student_kwargs(
        {
            "model": "student",
            "teacher_mode": "steer",
            "teacher_system_prompt_file": "teacher.txt",
            "teacher_system_prompt_placement": "prepend",
            "teacher_model": "teacher",
        }
    ) == {"model": "student"}


def test_generated_sources_are_followed_by_teacher_objective_only(tmp_path):
    prompt = tmp_path / "teacher.txt"
    prompt.write_text("Rewrite the objective.")
    objective_file = tmp_path / "objective.json"

    class FakeWorkflow:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            log_file = tmp_path / f"run-{len(self.instances)}.json"
            self.workflow_message = SimpleNamespace(log_file=log_file)
            self.instances.append(self)

        async def run(self):
            self.workflow_message.log_file.write_text(
                json.dumps(
                    {"workflow_metadata": {"workflow_summary": {"complete": True}}}
                )
            )

    runner = WorkflowRunner()
    runner._workflow_factory = {"detect_workflow": FakeWorkflow}
    runner.args = Namespace(
        workflow_type="detect_workflow",
        generate_source_runs=True,
        source_logs=None,
        teacher_system_prompt_file=str(prompt),
        teacher_system_prompt_placement="prepend",
        teacher_model="teacher",
        teacher_max_input_tokens=1000,
        teacher_max_output_tokens=100,
        teacher_temperature=0.0,
        use_mock_model=False,
        objective_output_dir=str(tmp_path),
        task_dir=str(tmp_path / "lunary"),
        bounty_number="0",
    )
    runner.kwargs = {
        "task_dir": tmp_path / "lunary",
        "bounty_number": "0",
        "model": "student",
        "teacher_mode": "objective_rewrite",
        "teacher_system_prompt_file": str(prompt),
        "teacher_system_prompt_placement": "prepend",
        "teacher_model": "teacher",
    }

    with (
        patch(
            "workflows.runner.rewrite_objective",
            AsyncMock(
                return_value=SimpleNamespace(
                    objective="Rewritten objective",
                    teacher_trace={"response": "Rewritten objective"},
                )
            ),
        ) as rewrite,
        patch("workflows.runner.write_objective", return_value=objective_file),
    ):
        asyncio.run(runner._run_objective_rewrite())

    assert len(FakeWorkflow.instances) == 3
    assert all("teacher_mode" not in run.kwargs for run in FakeWorkflow.instances)
    assert all(
        "objective_override" not in run.kwargs for run in FakeWorkflow.instances
    )
    assert len(rewrite.await_args.kwargs["source_logs"]) == 3
    assert rewrite.await_args.kwargs["system_prompt_placement"] is (
        TeacherSystemPromptPlacement.PREPEND
    )
