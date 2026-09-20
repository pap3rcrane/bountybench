#!/usr/bin/env python3
"""Render production-equivalent teacher inputs from three saved student traces."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

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
from workflows.teacher_modes import build_objective_rewrite_input  # noqa: E402

logging.disable(logging.CRITICAL)

DEFAULT_PLAN = REPOSITORY_ROOT / "matrix_inputs/user_vm4/task_designer_plan.jsonl"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs/teacher_input_examples"
TURN_HEADER = re.compile(r"TURN [1-9][0-9]*")


def load_plan_record(path: Path, line_number: int) -> dict:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if line_number < 1 or line_number > len(records):
        raise ValueError(
            f"--plan-line must be between 1 and {len(records)}, got {line_number}"
        )
    record = records[line_number - 1]
    source_logs = record.get("source_logs")
    if not isinstance(source_logs, list) or len(source_logs) != 3:
        raise ValueError("Selected plan record must contain exactly three source_logs")
    return record


def resolve_repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def history_from_source_log(path: Path) -> AgentMessage:
    payload = json.loads(path.read_text())
    previous: Optional[AgentMessage] = None

    for phase in payload.get("phase_messages", []):
        for raw_message in phase.get("agent_messages", []):
            message = AgentMessage(
                agent_id=raw_message.get("agent_id", "unknown_agent"),
                message=raw_message.get("message", ""),
                prev=previous,
            )
            iteration = raw_message.get("iteration")
            if iteration is not None:
                message.set_iteration(iteration)

            previous_action = None
            for raw_action in raw_message.get("action_messages") or []:
                action = ActionMessage(
                    resource_id=raw_action.get("resource_id", ""),
                    message=raw_action.get("message", ""),
                    additional_metadata=raw_action.get("additional_metadata"),
                    prev=previous_action,
                )
                message.add_child_message(action)
                previous_action = action
            previous = message

    if previous is None:
        raise ValueError(f"Source log contains no agent messages: {path}")
    return previous


def build_online_input(
    *,
    source_log: Path,
    mode: TeacherMode,
    prompt_file: Path,
    placement: TeacherSystemPromptPlacement,
) -> tuple[str, Optional[str]]:
    teacher = TeacherAgent(
        "teacher_agent",
        TeacherAgentConfig(
            system_prompt_file=str(prompt_file),
            system_prompt_placement=placement,
            mode=mode,
        ),
    )
    model_input = teacher.build_model_input(history_from_source_log(source_log))
    return model_input.memory, model_input.system_prompt


def turn_headers(contents: str) -> list[str]:
    headers = [line for line in contents.splitlines() if line.startswith("TURN ")]
    invalid = [header for header in headers if TURN_HEADER.fullmatch(header) is None]
    if invalid:
        raise ValueError(f"Unexpected turn header(s): {invalid[:3]}")
    return headers


def write_role_input(
    *,
    output_directory: Path,
    role: str,
    mode: TeacherMode,
    prompt_file: Path,
    placement: TeacherSystemPromptPlacement,
    source_logs: list[Path],
    contents: str,
    system_prompt: Optional[str],
) -> dict:
    user_input_path = output_directory / f"{role}_user_input.txt"
    system_prompt_path = output_directory / f"{role}_system_prompt.txt"
    user_input_path.write_text(contents)
    system_prompt_path.write_text(system_prompt or "")
    headers = turn_headers(contents)
    return {
        "role": role,
        "teacher_mode": mode.value,
        "teacher_system_prompt_placement": placement.value,
        "prompt_file": str(prompt_file.relative_to(REPOSITORY_ROOT)),
        "source_logs": [str(path.relative_to(REPOSITORY_ROOT)) for path in source_logs],
        "source_log_count": len(source_logs),
        "system_prompt_is_empty": system_prompt is None,
        "system_prompt_path": str(system_prompt_path.relative_to(REPOSITORY_ROOT)),
        "user_input_path": str(user_input_path.relative_to(REPOSITORY_ROOT)),
        "user_input_bytes": len(contents.encode("utf-8")),
        "turn_count": len(headers),
        "turn_headers": headers,
    }


def render_examples(plan: Path, line_number: int, output_directory: Path) -> dict:
    record = load_plan_record(plan, line_number)
    source_logs = [resolve_repository_path(value) for value in record["source_logs"]]
    placement = TeacherSystemPromptPlacement(record["system_prompt_placement"])
    task_designer_prompt = resolve_repository_path(record["system_prompt_file"])
    critiquer_prompt = resolve_repository_path(
        "prompts/system_prompts/single_error_correction_one_alternative.txt"
    )
    judge_prompt = resolve_repository_path(
        "prompts/system_prompts/outcome_evaluation.txt"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    critiquer_contents, critiquer_system = build_online_input(
        source_log=source_logs[0],
        mode=TeacherMode.OBSERVE,
        prompt_file=critiquer_prompt,
        placement=placement,
    )
    judge_contents, judge_system = build_online_input(
        source_log=source_logs[0],
        mode=TeacherMode.STEER,
        prompt_file=judge_prompt,
        placement=placement,
    )
    task_designer_input = build_objective_rewrite_input(
        task_designer_prompt,
        placement,
        source_logs,
    )

    roles = [
        write_role_input(
            output_directory=output_directory,
            role="critiquer",
            mode=TeacherMode.OBSERVE,
            prompt_file=critiquer_prompt,
            placement=placement,
            source_logs=source_logs[:1],
            contents=critiquer_contents,
            system_prompt=critiquer_system,
        ),
        write_role_input(
            output_directory=output_directory,
            role="judge_verifier",
            mode=TeacherMode.STEER,
            prompt_file=judge_prompt,
            placement=placement,
            source_logs=source_logs[:1],
            contents=judge_contents,
            system_prompt=judge_system,
        ),
        write_role_input(
            output_directory=output_directory,
            role="task_designer",
            mode=TeacherMode.OBJECTIVE_REWRITE,
            prompt_file=task_designer_prompt,
            placement=placement,
            source_logs=source_logs,
            contents=task_designer_input.memory,
            system_prompt=task_designer_input.system_prompt,
        ),
    ]
    manifest = {
        "plan": str(plan.relative_to(REPOSITORY_ROOT)),
        "plan_line": line_number,
        "configuration": {
            key: record[key]
            for key in (
                "repo_name",
                "bounty_number",
                "workflow_type",
                "system_prompt_name",
                "system_prompt_placement",
                "run_number",
            )
        },
        "note": (
            "Task Designer combines all three source traces. Online Critiquer and "
            "Judge/Verifier calls receive one trajectory, so their examples use "
            "the first source trace."
        ),
        "roles": roles,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--plan-line", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()

    plan = args.plan.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    manifest = render_examples(plan, args.plan_line, output_directory)
    print(f"Wrote teacher input examples to {output_directory}")
    for role in manifest["roles"]:
        print(
            f"{role['role']}: sources={role['source_log_count']} "
            f"turns={role['turn_count']} bytes={role['user_input_bytes']} "
            f"input={role['user_input_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
