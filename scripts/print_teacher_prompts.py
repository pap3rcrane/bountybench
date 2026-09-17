#!/usr/bin/env python3
"""Print complete Gemini prompt payloads for every teacher prompt option.

This is a dry test utility. It uses deterministic representative student traces and
the production teacher prompt builders, but it never initializes Docker and never
calls Gemini.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agents.teacher_agent import (  # noqa: E402
    TeacherAgent,
    TeacherAgentConfig,
    TeacherMode,
    TeacherSystemPromptPlacement,
)
from messages.action_messages.action_message import ActionMessage  # noqa: E402
from messages.agent_messages.agent_message import AgentMessage  # noqa: E402
from prompts.prompts import (  # noqa: E402
    DETECT_PROMPT,
    EXPLOIT_PROMPT,
    PATCH_PROMPT,
    STOP_TOKEN,
    SUBMISSION_INSTRUCTIONS,
)
from workflows.teacher_modes import build_objective_rewrite_input  # noqa: E402

# The preview itself is the output. Suppress normal benchmark logging so stdout can
# safely be redirected to a text file or parsed as JSON.
logging.disable(logging.CRITICAL)

GEMINI_MODEL = "google/gemini-3.6-flash"
DEFAULT_MAX_INPUT_TOKENS = 131072
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0

WORKFLOW_TEMPLATES = {
    "detect_workflow": DETECT_PROMPT,
    "exploit_workflow": EXPLOIT_PROMPT,
    "patch_workflow": PATCH_PROMPT,
}

PROMPTS_BY_MODE = {
    TeacherMode.OBSERVE: (
        "single_error_correction_one_alternative",
        "single_error_correction_multiple_alternatives",
        "optimizer",
        "principle_extraction",
        "student_nudging",
    ),
    TeacherMode.STEER: (
        "outcome_evaluation",
        "teacher_answer_comparison",
        "comparative_ranking",
        "reasoning_quality_fidelity",
        "rubric_based_evaluation",
    ),
    TeacherMode.OBJECTIVE_REWRITE: (
        "increased_difficulty_task",
        "weakness_targeting_task",
        "strategy_limiter_task",
        "justification_gap_task",
        "oversight_task",
    ),
}

PROMPT_MODE = {
    prompt_name: mode
    for mode, prompt_names in PROMPTS_BY_MODE.items()
    for prompt_name in prompt_names
}

CHECK_AGENT_BY_WORKFLOW = {
    "detect_workflow": "detect_agent",
    "exploit_workflow": "exploit_agent",
    "patch_workflow": "patch_agent",
}

STUDENT_ACTION_BY_WORKFLOW = {
    "detect_workflow": (
        "Thought: Inspect the codebase for a high-impact vulnerability.\n"
        'Command: rg -n "password|secret|token" codebase\n'
        f"{STOP_TOKEN}"
    ),
    "exploit_workflow": (
        "Thought: Reproduce the reported vulnerability before writing the exploit.\n"
        "Command: bash verify_files/verify.sh\n"
        f"{STOP_TOKEN}"
    ),
    "patch_workflow": (
        "Thought: Locate the vulnerable implementation before changing it.\n"
        'Command: rg -n "vulnerable_function" codebase\n'
        f"{STOP_TOKEN}"
    ),
}


def _prompt_path(prompt_name: str) -> Path:
    path = REPOSITORY_ROOT / "prompts" / "system_prompts" / f"{prompt_name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing teacher system prompt: {path}")
    return path


def _student_system_prompt(workflow_type: str) -> str:
    return WORKFLOW_TEMPLATES[workflow_type].format(
        task_description=(f"TEST FIXTURE: representative report for {workflow_type}."),
        info="TEST FIXTURE: representative repository metadata.",
        codebase="$ tree -L 2\n.\n└── codebase\n    └── example.py",
        specialized_instructions="",
        submit=SUBMISSION_INSTRUCTIONS,
        stop_token=STOP_TOKEN,
        target_host="test-app:8000",
        bounty_setup_hosts="",
        resolved_host="test-app:8000",
        cwe="CWE-000",
    )


def _online_history(workflow_type: str) -> AgentMessage:
    system = AgentMessage(
        agent_id="system", message=_student_system_prompt(workflow_type)
    )
    system.set_iteration(-1)

    student = AgentMessage(agent_id="executor_agent", prev=system)
    student.set_iteration(0)
    model_action = ActionMessage(
        resource_id="model", message=STUDENT_ACTION_BY_WORKFLOW[workflow_type]
    )
    student.add_child_message(model_action)
    student.add_child_message(
        ActionMessage(
            resource_id="kali_env_test",
            message=(
                "TEST FIXTURE: representative environment output for "
                f"{workflow_type}."
            ),
            prev=model_action,
        )
    )

    evaluator = AgentMessage(
        agent_id=CHECK_AGENT_BY_WORKFLOW[workflow_type],
        message="TEST FIXTURE: evaluator has inspected the student turn.",
        prev=student,
    )
    evaluator.set_iteration(1)
    return evaluator


def _source_log_payload(workflow_type: str, run_number: int) -> dict:
    return {
        "workflow_metadata": {
            "workflow_name": workflow_type,
            "task": "teacher prompt preview fixture",
        },
        "phase_messages": [
            {
                "phase_id": CHECK_AGENT_BY_WORKFLOW[workflow_type],
                "phase_summary": "no_submission/failure",
                "success": False,
                "submit": False,
                "agent_messages": [
                    {
                        "agent_id": "system",
                        "message": _student_system_prompt(workflow_type),
                    },
                    {
                        "agent_id": "executor_agent",
                        "message": "",
                        "action_messages": [
                            {
                                "resource_id": "model",
                                "message": (
                                    f"Representative student run {run_number}.\n"
                                    f"{STUDENT_ACTION_BY_WORKFLOW[workflow_type]}"
                                ),
                            },
                            {
                                "resource_id": "kali_env_test",
                                "message": (
                                    "TEST FIXTURE: representative environment output "
                                    f"from source run {run_number}."
                                ),
                            },
                        ],
                    },
                    {
                        "agent_id": CHECK_AGENT_BY_WORKFLOW[workflow_type],
                        "message": "TEST FIXTURE: no final submission yet.",
                    },
                ],
            }
        ],
    }


def _write_source_logs(directory: Path, workflow_type: str) -> list[Path]:
    workflow_dir = directory / workflow_type
    workflow_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_number in range(1, 4):
        path = workflow_dir / f"source_run_{run_number}.json"
        path.write_text(
            json.dumps(_source_log_payload(workflow_type, run_number), indent=2) + "\n"
        )
        paths.append(path)
    return paths


def _gemini_request(
    *,
    workflow_type: str,
    prompt_name: Optional[str],
    mode: TeacherMode,
    placement: TeacherSystemPromptPlacement,
    source_logs: list[Path],
    max_input_tokens: int,
    max_output_tokens: int,
    temperature: float,
) -> dict:
    prompt_file = _prompt_path(prompt_name) if prompt_name else None

    if mode is TeacherMode.OBJECTIVE_REWRITE:
        model_input = build_objective_rewrite_input(
            prompt_file, placement, source_logs
        )
        contents = model_input.memory
        system_instruction = model_input.system_prompt
    else:
        teacher = TeacherAgent(
            "teacher_agent",
            TeacherAgentConfig(
                system_prompt_file=str(prompt_file) if prompt_file else None,
                system_prompt_placement=placement,
                mode=mode,
            ),
        )
        model_input = teacher.build_model_input(_online_history(workflow_type))
        contents = model_input.memory
        system_instruction = model_input.system_prompt

    # These deterministic fixtures are far below the default input limit, so the
    # production ModelResource passes this contents string through unchanged.
    if len(contents.encode("utf-8")) >= max_input_tokens:
        raise ValueError(
            "The fixture may exceed --max-input-tokens. Increase the limit so the "
            "preview remains an exact, untruncated Gemini request."
        )
    return {
        "workflow_type": workflow_type,
        "teacher_mode": mode.value,
        "system_prompt_name": prompt_name,
        "system_prompt_file": str(prompt_file) if prompt_file else None,
        "teacher_system_prompt_placement": placement.value,
        "gemini_api": {
            "GenerativeModel": {
                "model_id": GEMINI_MODEL.split("/", 1)[-1],
                "system_instruction": system_instruction,
            },
            "generate_content": {
                "contents": contents,
                "generation_config": {
                    "temperature": temperature,
                    "stop_sequences": [],
                    "max_output_tokens": max_output_tokens,
                },
            },
        },
    }


def build_requests(
    *,
    workflow_types: Iterable[str],
    prompt_names: Iterable[str],
    placements: Iterable[TeacherSystemPromptPlacement],
    fixture_directory: Path,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict]:
    prompt_names = list(prompt_names)
    placements = list(placements)
    requests = []
    for workflow_type in workflow_types:
        source_logs = _write_source_logs(fixture_directory, workflow_type)
        for mode, mode_prompt_names in PROMPTS_BY_MODE.items():
            selected_prompt_names = [
                prompt_name
                for prompt_name in prompt_names
                if prompt_name in mode_prompt_names
            ]
            if not selected_prompt_names:
                continue

            # `none` is one no-custom-prompt baseline per selected teacher mode,
            # rather than a duplicate baseline for each prompt file in that mode.
            if TeacherSystemPromptPlacement.NONE in placements:
                requests.append(
                    _gemini_request(
                        workflow_type=workflow_type,
                        prompt_name=None,
                        mode=mode,
                        placement=TeacherSystemPromptPlacement.NONE,
                        source_logs=source_logs,
                        max_input_tokens=max_input_tokens,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                )

            for prompt_name in selected_prompt_names:
                for placement in placements:
                    if placement is TeacherSystemPromptPlacement.NONE:
                        continue
                    requests.append(
                        _gemini_request(
                            workflow_type=workflow_type,
                            prompt_name=prompt_name,
                            mode=mode,
                            placement=placement,
                            source_logs=source_logs,
                            max_input_tokens=max_input_tokens,
                            max_output_tokens=max_output_tokens,
                            temperature=temperature,
                        )
                    )
    return requests


def _render_text(requests: list[dict]) -> str:
    sections = [
        "TEST FIXTURE: deterministic representative traces are used; real Gemini "
        "contents change with the complete student trace.\n\n"
        "PLACEMENT OPTIONS\n"
        "- none: no custom prompt file; task and trace are sent as contents and "
        "system_instruction is empty.\n"
        "- prepend: custom prompt, task, and trace are combined in contents; "
        "system_instruction is empty.\n"
        "- system: custom prompt and task are sent in Gemini system_instruction; "
        "contents contains only the trace.\n\n"
        "PROMPT OPTIONS BY TEACHER MODE\n"
        + "\n".join(
            f"- {mode.value}: {', '.join(prompt_names)}"
            for mode, prompt_names in PROMPTS_BY_MODE.items()
        )
    ]
    for number, request in enumerate(requests, start=1):
        api = request["gemini_api"]
        system_instruction = api["GenerativeModel"]["system_instruction"]
        sections.append(
            "\n".join(
                (
                    "=" * 100,
                    f"REQUEST {number}/{len(requests)}",
                    f"workflow_type: {request['workflow_type']}",
                    f"teacher_mode: {request['teacher_mode']}",
                    f"system_prompt_name: {request['system_prompt_name']}",
                    (
                        "teacher_system_prompt_placement: "
                        f"{request['teacher_system_prompt_placement']}"
                    ),
                    f"Gemini model_id: {api['GenerativeModel']['model_id']}",
                    "-" * 100,
                    "Gemini system_instruction:",
                    system_instruction if system_instruction is not None else "<NONE>",
                    "-" * 100,
                    "Gemini contents:",
                    api["generate_content"]["contents"],
                    "-" * 100,
                    "Gemini generation_config:",
                    json.dumps(api["generate_content"]["generation_config"], indent=2),
                )
            )
        )
    return "\n\n".join(sections) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the complete prompt fields that would be sent to Gemini for "
            "teacher workflow/system-prompt pairs. No API call is made."
        )
    )
    parser.add_argument(
        "--workflow",
        action="append",
        choices=tuple(WORKFLOW_TEMPLATES),
        help="Workflow to include; repeat this flag. Default: all three.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        choices=tuple(PROMPT_MODE),
        help="System prompt name without .txt; repeat this flag. Default: all 15.",
    )
    parser.add_argument(
        "--placement",
        action="append",
        choices=(
            TeacherSystemPromptPlacement.NONE.value,
            TeacherSystemPromptPlacement.PREPEND.value,
            TeacherSystemPromptPlacement.SYSTEM.value,
        ),
        help="Prompt placement to include; repeat this flag. Default: all three.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to this file instead of stdout.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=(
            "Require each fixture to fit below this production input limit "
            "(default: 131072)."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Value shown in Gemini generation_config.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Value shown in Gemini generation_config.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    workflow_types = args.workflow or list(WORKFLOW_TEMPLATES)
    prompt_names = args.prompt or list(PROMPT_MODE)
    placements = [
        TeacherSystemPromptPlacement(value)
        for value in (
            args.placement
            or (
                TeacherSystemPromptPlacement.NONE.value,
                TeacherSystemPromptPlacement.PREPEND.value,
                TeacherSystemPromptPlacement.SYSTEM.value,
            )
        )
    ]

    if args.max_input_tokens <= 0:
        raise SystemExit("--max-input-tokens must be positive")
    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be positive")

    with tempfile.TemporaryDirectory(prefix="bountybench-teacher-prompt-test-") as tmp:
        requests = build_requests(
            workflow_types=workflow_types,
            prompt_names=prompt_names,
            placements=placements,
            fixture_directory=Path(tmp),
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
        if args.format == "json":
            output = (
                json.dumps(
                    {
                        "test_fixture": (
                            "Deterministic representative traces; real run contents "
                            "change with the complete student trace."
                        ),
                        "request_count": len(requests),
                        "requests": requests,
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            output = _render_text(requests)

        if args.output:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output)
            print(
                f"Wrote {len(requests)} complete Gemini prompt payloads to {output_path}"
            )
        else:
            sys.stdout.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
