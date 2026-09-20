from unittest.mock import patch

import pytest

from messages.action_messages.action_message import ActionMessage
from messages.agent_messages.agent_message import AgentMessage
from messages.phase_messages.phase_message import PhaseMessage
from messages.workflow_message import WorkflowMessage
from prompts.teacher_response_context import format_teacher_output_for_student
from resources.memory_resource.memory_function import MemoryTruncationFunctions
from resources.memory_resource.memory_resource import (
    MemoryResource,
    MemoryResourceConfig,
)


@pytest.fixture(autouse=True)
def disable_workflow_saves():
    with patch.object(WorkflowMessage, "save", return_value=None):
        yield


def build_teacher_message(response):
    workflow = WorkflowMessage("")
    phase = PhaseMessage("test_phase")
    workflow.add_child_message(phase)

    system = AgentMessage("system", "initial prompt")
    phase.add_child_message(system)
    student = AgentMessage("executor_agent", prev=system)
    student.add_child_message(ActionMessage("model", "student action"))
    phase.add_child_message(student)
    evaluator = AgentMessage("detect_agent", "evaluation", prev=student)
    phase.add_child_message(evaluator)
    teacher = AgentMessage(
        "teacher_agent",
        format_teacher_output_for_student(
            "observe",
            "prompts/system_prompts/single_error_correction_one_alternative.txt",
            response,
        ),
        prev=evaluator,
    )
    phase.add_child_message(teacher)
    return teacher


def memory_resource():
    return MemoryResource(
        "memory",
        MemoryResourceConfig(
            segment_trunc_fn=MemoryTruncationFunctions.segment_fn_noop,
        ),
    )


def test_latest_teacher_response_is_shortened_then_protected():
    response = " ".join(f"teacher-word-{i}" for i in range(1_000))
    teacher = build_teacher_message(response)

    memory = memory_resource().get_memory(teacher).memory

    assert teacher.message not in memory
    assert "...TRUNCATED..." in memory
    assert teacher.protected_memory_content is not None
    assert teacher.required_memory_prefix == "initial prompt"
    assert memory.endswith(teacher.protected_memory_content)


def test_older_teacher_response_uses_normal_per_message_shortening():
    response = " ".join(f"teacher-word-{i}" for i in range(1_000))
    teacher = build_teacher_message(response)
    later_agent = AgentMessage("detect_agent", "later evaluation", prev=teacher)
    teacher.parent.add_child_message(later_agent)

    memory = memory_resource().get_memory(later_agent).memory

    assert teacher.message not in memory
    assert "...TRUNCATED..." in memory
    assert later_agent.protected_memory_content is None
    assert later_agent.required_memory_prefix is None
