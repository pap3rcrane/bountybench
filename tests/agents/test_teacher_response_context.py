import re
from pathlib import Path

import pytest

from messages.action_messages.action_message import ActionMessage
from messages.agent_messages.agent_message import AgentMessage
from messages.phase_messages.phase_message import PhaseMessage
from messages.workflow_message import WorkflowMessage
from prompts.teacher_response_context import (
    BASELINE_TEACHER_STUDENT_CONTEXT,
    TEACHER_STUDENT_CONTEXT,
    format_teacher_output_for_student,
)
from resources.memory_resource.memory_function import MemoryTruncationFunctions
from resources.memory_resource.memory_resource import (
    MemoryResource,
    MemoryResourceConfig,
)


PASS_NO_INPUT_CONTEXT = "If the teacher returns PASS, it has no input."


def test_pass_system_prompts_append_no_input_context():
    prompt_dir = Path(__file__).resolve().parents[2] / "prompts" / "system_prompts"
    pass_prompt_names = {
        prompt_path.stem
        for prompt_path in prompt_dir.glob("*.txt")
        if re.search(r"\bPASS\b", prompt_path.read_text())
    }
    prefixed_prompt_names = {
        prompt_name
        for mode_contexts in TEACHER_STUDENT_CONTEXT.values()
        for prompt_name, context in mode_contexts.items()
        if context.endswith(PASS_NO_INPUT_CONTEXT)
    }

    assert pass_prompt_names == prefixed_prompt_names


@pytest.mark.parametrize("mode", ["observe", "steer"])
def test_no_prompt_teacher_output_has_no_prefix_and_reaches_student_memory(mode):
    response = "Useful critique."
    teacher_output = format_teacher_output_for_student(mode, None, response)

    assert BASELINE_TEACHER_STUDENT_CONTEXT[mode] == ""
    assert teacher_output == response

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
    teacher = AgentMessage("teacher_agent", teacher_output, prev=evaluator)
    phase.add_child_message(teacher)

    memory = MemoryResource(
        "memory",
        MemoryResourceConfig(
            collate_fn=lambda messages, start=0: " ".join(messages),
            segment_trunc_fn=MemoryTruncationFunctions.segment_fn_noop,
            memory_trunc_fn=MemoryTruncationFunctions.memory_fn_noop,
        ),
    ).get_memory(teacher).memory

    assert f"[teacher_agent] {response}" in memory
