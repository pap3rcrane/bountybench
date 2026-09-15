from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from agents.base_agent import AgentConfig, BaseAgent
from messages.agent_messages.agent_message import AgentMessage
from messages.message import Message
from resources.resource_type import ResourceType
from utils.logger import get_main_logger

logger = get_main_logger(__name__)


def format_original_task(original_task: str) -> str:
    return (
        "ORIGINAL BENCHMARK TASK:\n"
        "---\n"
        f"{original_task.strip()}\n"
        "---\n"
    )


def format_teacher_trace(trace: str) -> str:
    return (
        "AVAILABLE TRACE (oldest to newest):\n"
        "---\n"
        f"{trace}\n"
        "---\n"
    )


def format_teacher_context(original_task: str, trace: str) -> str:
    return f"{format_original_task(original_task)}\n{format_teacher_trace(trace)}"


class TeacherMode(str, Enum):
    OBSERVE = "observe"
    STEER = "steer"
    OBJECTIVE_REWRITE = "objective_rewrite"


class TeacherSystemPromptPlacement(str, Enum):
    NONE = "none"
    PREPEND = "prepend"
    SYSTEM = "system"


@dataclass(frozen=True)
class TeacherAgentConfig(AgentConfig):
    system_prompt_file: Optional[str]
    system_prompt_placement: TeacherSystemPromptPlacement
    mode: TeacherMode


@dataclass
class _TeacherModelInput:
    memory: str
    system_prompt: Optional[str] = None
    message: str = "Teacher mock response."
    trace_turns: int = 0


class TeacherAgent(BaseAgent):
    """A critique-only companion scheduled after the benchmark evaluator."""

    REQUIRED_RESOURCES = [ResourceType.TEACHER_MODEL]
    OPTIONAL_RESOURCES = []
    ACCESSIBLE_RESOURCES = [ResourceType.TEACHER_MODEL]

    def __init__(self, agent_id: str, agent_config: TeacherAgentConfig):
        super().__init__(agent_id, agent_config)
        self.system_prompt_placement = TeacherSystemPromptPlacement(
            agent_config.system_prompt_placement
        )
        if self.system_prompt_placement is TeacherSystemPromptPlacement.NONE:
            if agent_config.system_prompt_file is not None:
                raise ValueError(
                    "Teacher system prompt file must be omitted when placement is none"
                )
            self.system_prompt_path = None
        else:
            if agent_config.system_prompt_file is None:
                raise ValueError(
                    "Teacher system prompt file is required for prepend and system "
                    "placement"
                )
            self.system_prompt_path = self._resolve_prompt_path(
                agent_config.system_prompt_file
            )
        self.mode = TeacherMode(agent_config.mode)
        if self.mode is TeacherMode.OBJECTIVE_REWRITE:
            raise ValueError(
                "objective_rewrite is an offline teacher mode and cannot be "
                "scheduled inside a phase"
            )

    @staticmethod
    def _resolve_prompt_path(prompt_file: str) -> Path:
        path = Path(prompt_file).expanduser()
        if not path.is_absolute():
            repository_root = Path(__file__).resolve().parents[2]
            path = repository_root / path
        if not path.is_file():
            raise ValueError(f"Teacher system prompt file does not exist: {path}")
        return path

    @staticmethod
    def build_trace(latest_message: Message) -> str:
        """Serialize the student-only portion of the linked history."""
        history = []
        seen = set()
        current = latest_message
        while current is not None and current.id not in seen:
            seen.add(current.id)
            history.append(current.get_latest_version())
            current = current.prev

        lines = []
        turn = 0
        for message in reversed(history):
            agent_id = getattr(message, "agent_id", message.message_type)
            if agent_id != "executor_agent":
                continue
            turn += 1
            iteration = getattr(message, "iteration", None)
            iteration_label = "" if iteration is None else f" iteration={iteration}"
            lines.append(f"TURN {turn} agent={agent_id}{iteration_label}")

            message_text = getattr(message, "message", "")
            if message_text:
                lines.append(message_text)

            actions = getattr(message, "current_children", [])
            for action in actions:
                lines.append(f"[{action.resource_id}]")
                lines.append(action.message)

        return "\n\n".join(lines)

    @staticmethod
    def find_original_task(latest_message: Message) -> str:
        original_task = None
        seen = set()
        current = latest_message
        while current is not None and current.id not in seen:
            seen.add(current.id)
            message = current.get_latest_version()
            if getattr(message, "agent_id", message.message_type) == "system":
                original_task = message.message
            current = current.prev
        if original_task is None:
            raise ValueError("Teacher trace is missing the original benchmark task")
        return original_task

    def _build_user_prompt(self, original_task: str, trace: str) -> str:
        visibility = (
            "Your response is private and will not be shown to the student."
            if self.mode is TeacherMode.OBSERVE
            else "Your response will be shown to the student before its next turn."
        )
        if self.system_prompt_placement is TeacherSystemPromptPlacement.SYSTEM:
            return format_teacher_trace(trace)

        user_prompt = format_teacher_context(original_task, trace)
        if self.system_prompt_placement is TeacherSystemPromptPlacement.PREPEND:
            assert self.system_prompt_path is not None
            system_prompt = self.system_prompt_path.read_text().strip()
            return f"{system_prompt}\n\n{user_prompt}"
        return user_prompt

    def build_model_input(self, previous: Message) -> _TeacherModelInput:
        """Build the exact prompt fields passed to the teacher model resource."""
        trace = self.build_trace(previous)
        original_task = self.find_original_task(previous)
        return _TeacherModelInput(
            memory=self._build_user_prompt(original_task, trace),
            system_prompt=(
                f"{self.system_prompt_path.read_text().strip()}\n\n"
                f"{format_original_task(original_task)}"
                if self.system_prompt_placement is TeacherSystemPromptPlacement.SYSTEM
                and self.system_prompt_path is not None
                else None
            ),
            trace_turns=trace.count("TURN "),
        )

    async def run(self, messages: List[AgentMessage]) -> AgentMessage:
        if len(messages) != 1:
            raise ValueError(
                f"TeacherAgent accepts one previous message, received {len(messages)}."
            )

        previous = messages[0]
        teacher_message = AgentMessage(agent_id=self.agent_id, prev=previous)
        model_input = self.build_model_input(previous)
        teacher_action = await asyncio.to_thread(
            self.resources.teacher_model.run, model_input
        )
        teacher_action.add_to_additional_metadata(
            "teacher",
            {
                "teacher_system_prompt_file": (
                    str(self.system_prompt_path) if self.system_prompt_path else None
                ),
                "teacher_system_prompt_placement": self.system_prompt_placement.value,
                "mode": self.mode.value,
                "environment_access": False,
                "trace_turns": model_input.trace_turns,
            },
        )
        teacher_message.add_child_message(teacher_action)
        logger.info(
            "\n%s\nTEACHER RESPONSE (%s)\n%s\n%s",
            "=" * 80,
            self.mode.value,
            teacher_action.message,
            "=" * 80,
        )
        if self.mode is TeacherMode.STEER:
            teacher_message.set_message(f"Teacher response:\n{teacher_action.message}")
        return teacher_message

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "teacher_system_prompt_file": (
                str(self.system_prompt_path) if self.system_prompt_path else None
            ),
            "teacher_system_prompt_placement": self.system_prompt_placement.value,
            "mode": self.mode.value,
            "environment_access": False,
        }
