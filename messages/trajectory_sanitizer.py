"""Produce analysis-ready trajectories without usage metadata."""

from __future__ import annotations

import re
from typing import Any


OMITTED_TRAJECTORY_KEYS = {"workflow_usage", "phase_usage", "resources_used"}


def _teacher_mode(agent_message: dict[str, Any]) -> str | None:
    for action in agent_message.get("action_messages") or []:
        metadata = action.get("additional_metadata") or {}
        teacher = metadata.get("teacher") or {}
        if teacher.get("mode"):
            return str(teacher["mode"])
    return None


def _teacher_response(agent_message: dict[str, Any]) -> str:
    for action in agent_message.get("action_messages") or []:
        metadata = action.get("additional_metadata") or {}
        response = action.get("message")
        if (
            metadata.get("teacher")
            and isinstance(response, str)
            and response.strip()
        ):
            return response
    return ""


def _next_student_input(
    agent_messages: list[Any], start: int
) -> dict[str, Any] | None:
    for agent_message in agent_messages[start:]:
        if not isinstance(agent_message, dict):
            continue
        if agent_message.get("agent_id") == "teacher_agent":
            return None
        if agent_message.get("agent_id") != "executor_agent":
            continue
        for action in agent_message.get("action_messages") or []:
            metadata = action.get("additional_metadata") or {}
            if isinstance(metadata.get("input"), str):
                return metadata
        return None
    return None


def _expose_observe_feedback(agent_messages: list[Any]) -> None:
    """Make historical observe feedback use the same visible shape as steer.

    Older observe trajectories recorded the teacher model action but left the
    enclosing teacher message empty, so the following student's saved input did
    not contain the feedback. This changes only the exported copy.
    """
    for index, agent_message in enumerate(agent_messages):
        if not isinstance(agent_message, dict):
            continue
        if agent_message.get("agent_id") != "teacher_agent":
            continue
        if _teacher_mode(agent_message) != "observe":
            continue

        response = _teacher_response(agent_message)
        if not response:
            continue
        teacher_message = agent_message.get("message") or (
            f"Teacher response:\n{response}"
        )

        student_metadata = _next_student_input(agent_messages, index + 1)
        if student_metadata is None:
            continue
        student_input = student_metadata["input"]
        if f") [teacher_agent] {teacher_message}" in student_input:
            continue

        # Reconstruct only feedback that was originally invisible. Do not
        # rewrite already-visible turns or incomplete turns with no next input.
        agent_message["message"] = teacher_message
        numbered_messages = re.findall(r"(?m)^(\d+)\) \[", student_input)
        next_number = (int(numbered_messages[-1]) if numbered_messages else 0) + 1
        student_metadata["input"] = (
            f"{student_input.rstrip()}\n"
            f"{next_number}) [teacher_agent] {teacher_message}"
        )


def sanitize_trajectory(
    value: Any, *, expose_observe_feedback: bool = False
) -> Any:
    """Remove usage fields and optionally expose old observe feedback."""
    if isinstance(value, list):
        return [
            sanitize_trajectory(
                item, expose_observe_feedback=expose_observe_feedback
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    sanitized = {
        key: sanitize_trajectory(
            item, expose_observe_feedback=expose_observe_feedback
        )
        for key, item in value.items()
        if key not in OMITTED_TRAJECTORY_KEYS and "token" not in key.lower()
    }
    if expose_observe_feedback and isinstance(sanitized.get("agent_messages"), list):
        _expose_observe_feedback(sanitized["agent_messages"])
    return sanitized
