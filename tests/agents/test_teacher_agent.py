import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.detect_agent.detect_agent import DetectAgent
from agents.executor_agent.executor_agent import ExecutorAgent
from agents.teacher_agent import (
    TeacherAgent,
    TeacherAgentConfig,
    TeacherMode,
    TeacherSystemPromptPlacement,
)
from messages.action_messages.action_message import ActionMessage
from messages.agent_messages.agent_message import AgentMessage
from phases.detect_patch_phase import DetectPatchPhase
from phases.detect_phase import DetectPhase
from phases.exploit_phase import ExploitPhase
from phases.patch_phase import PatchPhase
from prompts.teacher_response_context import (
    BASELINE_TEACHER_STUDENT_CONTEXT,
    TEACHER_STUDENT_CONTEXT,
    format_teacher_output_for_student,
)
from resources.model_resource.model_mapping import get_model_info
from resources.model_resource.services.service_providers import ServiceProvider
from resources.resource_type import ResourceType


class FakeTeacherModel:
    resource_id = "teacher_model"

    def __init__(self, response):
        self.response = response
        self.inputs = []

    def run(self, input_message):
        self.inputs.append(input_message)
        return ActionMessage(resource_id=self.resource_id, message=self.response)


def make_history():
    system = AgentMessage(agent_id="system", message="original task")
    system.set_iteration(-1)
    executor = AgentMessage(agent_id="executor_agent", prev=system)
    executor.set_iteration(0)
    model_action = ActionMessage(
        resource_id="model",
        message="Thought: inspect files\nCommand: ls\n<END>",
    )
    executor.add_child_message(model_action)
    executor.add_child_message(
        ActionMessage(
            resource_id="kali_env_test", message="file-a\nfile-b", prev=model_action
        )
    )
    evaluator = AgentMessage(
        agent_id="detect_agent", message="Waiting for submission.", prev=executor
    )
    evaluator.set_iteration(1)
    old_teacher = AgentMessage(
        agent_id="teacher_agent", message="previous teacher observation", prev=evaluator
    )
    old_teacher.set_iteration(2)
    second_executor = AgentMessage(agent_id="executor_agent", prev=old_teacher)
    second_executor.set_iteration(3)
    second_executor.add_child_message(
        ActionMessage(resource_id="model", message="Command: pwd\n<END>")
    )
    second_evaluator = AgentMessage(
        agent_id="detect_agent", message="second evaluator result", prev=second_executor
    )
    second_evaluator.set_iteration(4)
    return second_evaluator


def make_agent(
    tmp_path,
    *,
    mode,
    placement=TeacherSystemPromptPlacement.PREPEND,
    response="Useful critique.",
    prompt_name=None,
):
    if prompt_name is None:
        prompt_name = (
            "single_error_correction_one_alternative"
            if mode is TeacherMode.OBSERVE
            else "outcome_evaluation"
        )
    prompt = tmp_path / f"{prompt_name}.txt"
    prompt.write_text("CUSTOM TEACHER SYSTEM PROMPT")
    agent = TeacherAgent(
        "teacher_agent",
        TeacherAgentConfig(
            system_prompt_file=(
                None if placement is TeacherSystemPromptPlacement.NONE else str(prompt)
            ),
            system_prompt_placement=placement,
            mode=mode,
        ),
    )
    agent.resources.teacher_model = FakeTeacherModel(response)
    return agent


def make_detect_phase(tmp_path, teacher_mode=None):
    (tmp_path / "bounties" / "bounty_0").mkdir(parents=True, exist_ok=True)
    workflow = MagicMock()
    workflow.params = {
        "use_mock_model": True,
        "teacher_model": "mock-teacher",
        "teacher_max_input_tokens": 1048576,
        "teacher_max_output_tokens": 65536,
        "teacher_temperature": 0.0,
    }
    if teacher_mode:
        workflow.params.update(
            {
                "teacher_mode": teacher_mode,
                "teacher_system_prompt_file": (
                    "prompts/teacher_agent_system_prompt.txt"
                ),
                "teacher_system_prompt_placement": "prepend",
            }
        )
    workflow.bounty_metadata = {"compatible_exploits": ["0"]}
    workflow.repo_metadata = {"target_host": "target:8080"}
    workflow.task_dir = tmp_path
    workflow.bounty_number = "0"
    workflow.workflow_message.workflow_id = "test-workflow"
    return DetectPhase(
        workflow=workflow,
        bounty_number="0",
        model="mock-student",
        use_mock_model=True,
        max_iterations=3,
        initial_prompt="test prompt",
    )


def test_teacher_is_a_native_phase_agent():
    for phase_class in (DetectPhase, ExploitPhase, PatchPhase, DetectPatchPhase):
        assert TeacherAgent in phase_class.AGENT_CLASSES


def test_baseline_does_not_schedule_teacher(tmp_path):
    phase = make_detect_phase(tmp_path)
    assert list(phase.define_agents()) == ["executor_agent", "detect_agent"]
    assert ResourceType.TEACHER_MODEL not in [
        resource for resource, _ in phase.define_resources()
    ]


def test_observe_schedules_teacher_after_native_agents(tmp_path):
    phase = make_detect_phase(tmp_path, TeacherMode.OBSERVE.value)
    assert list(phase.define_agents()) == [
        "executor_agent",
        "detect_agent",
        "teacher_agent",
    ]
    assert ResourceType.TEACHER_MODEL in [
        resource for resource, _ in phase.define_resources()
    ]
    teacher_config = next(
        config
        for resource, config in phase.define_resources()
        if resource is ResourceType.TEACHER_MODEL
    )
    assert teacher_config.preserve_oldest_input is False
    assert teacher_config.preserve_newest_input is True

    phase.agents = [
        ("executor_agent", object.__new__(ExecutorAgent)),
        ("detect_agent", object.__new__(DetectAgent)),
        ("teacher_agent", object.__new__(TeacherAgent)),
    ]
    check_agent_id, _ = phase._get_current_agent(get_prev=True)
    assert check_agent_id == "detect_agent"


def test_direct_gemini_teacher_uses_high_thinking(tmp_path):
    phase = make_detect_phase(tmp_path, TeacherMode.OBSERVE.value)
    phase.workflow.params["teacher_model"] = "google/gemini-3.6-flash"

    assert phase.teacher_model_config().thinking_level == "high"


def test_observe_sees_only_student_trace_and_exposes_response(tmp_path):
    agent = make_agent(tmp_path, mode=TeacherMode.OBSERVE)
    result = asyncio.run(agent.run([make_history()]))

    teacher_input = agent.resources.teacher_model.inputs[0].memory
    assert "CUSTOM TEACHER SYSTEM PROMPT" in teacher_input
    assert "Command: ls" in teacher_input
    assert "file-a\nfile-b" in teacher_input
    assert "Command: pwd" in teacher_input
    assert "original task" in teacher_input
    assert "evaluator result" not in teacher_input
    assert "previous teacher observation" not in teacher_input
    assert teacher_input.index("ORIGINAL BENCHMARK TASK:") < teacher_input.index(
        "AVAILABLE TRACE (oldest to newest):"
    )
    assert [
        line for line in teacher_input.splitlines() if line.startswith("TURN ")
    ] == [
        "TURN 1",
        "TURN 2",
    ]
    assert result.message == format_teacher_output_for_student(
        "observe", agent.system_prompt_path, "Useful critique."
    )
    assert result.action_messages[0].message == "Useful critique."
    assert result.action_messages[0].additional_metadata["teacher"]["mode"] == (
        "observe"
    )
    assert (
        result.action_messages[0].additional_metadata["teacher"][
            "teacher_system_prompt_placement"
        ]
        == "prepend"
    )


def test_system_placement_separates_system_prompt_from_user_prompt(tmp_path):
    agent = make_agent(
        tmp_path,
        mode=TeacherMode.OBSERVE,
        placement=TeacherSystemPromptPlacement.SYSTEM,
    )
    asyncio.run(agent.run([make_history()]))

    teacher_input = agent.resources.teacher_model.inputs[0]
    assert teacher_input.system_prompt == "CUSTOM TEACHER SYSTEM PROMPT"
    assert "ORIGINAL BENCHMARK TASK" not in teacher_input.system_prompt
    assert "CUSTOM TEACHER SYSTEM PROMPT" not in teacher_input.memory
    assert "ORIGINAL BENCHMARK TASK:\n---\noriginal task\n---" in teacher_input.memory
    assert "AVAILABLE TRACE" in teacher_input.memory


@pytest.mark.parametrize("mode", [TeacherMode.OBSERVE, TeacherMode.STEER])
def test_none_placement_passes_teacher_response_without_custom_prompt(tmp_path, mode):
    agent = make_agent(
        tmp_path,
        mode=mode,
        placement=TeacherSystemPromptPlacement.NONE,
    )
    result = asyncio.run(agent.run([make_history()]))

    teacher_input = agent.resources.teacher_model.inputs[0]
    assert teacher_input.system_prompt is None
    assert "CUSTOM TEACHER SYSTEM PROMPT" not in teacher_input.memory
    assert "AVAILABLE TRACE" in teacher_input.memory
    assert result.message == format_teacher_output_for_student(
        mode.value, None, "Useful critique."
    )
    assert BASELINE_TEACHER_STUDENT_CONTEXT[mode.value] == ""
    assert result.message == "Useful critique."
    assert (
        result.action_messages[0].additional_metadata["teacher"][
            "teacher_system_prompt_file"
        ]
        is None
    )


def test_steer_uses_observe_input_shape_and_exposes_response(tmp_path):
    agent = make_agent(tmp_path, mode=TeacherMode.STEER)
    result = asyncio.run(agent.run([make_history()]))

    teacher_input = agent.resources.teacher_model.inputs[0].memory
    assert "original task" in teacher_input
    assert "second evaluator result" not in teacher_input
    assert "previous teacher observation" not in teacher_input
    assert [
        line for line in teacher_input.splitlines() if line.startswith("TURN ")
    ] == [
        "TURN 1",
        "TURN 2",
    ]
    assert result.message == format_teacher_output_for_student(
        "steer", agent.system_prompt_path, "Useful critique."
    )


def test_every_matrix_online_teacher_prompt_has_student_context():
    assert set(TEACHER_STUDENT_CONTEXT["observe"]) == {
        "single_error_correction_one_alternative",
        "single_error_correction_multiple_alternatives",
        "optimizer",
        "principle_extraction",
        "student_nudging",
    }
    assert set(TEACHER_STUDENT_CONTEXT["steer"]) == {
        "outcome_evaluation",
        "teacher_answer_comparison",
        "comparative_ranking",
        "reasoning_quality_fidelity",
        "rubric_based_evaluation",
    }


def test_pending_student_turn_is_evaluated_then_reviewed(tmp_path):
    phase = make_detect_phase(tmp_path, TeacherMode.OBSERVE.value)
    phase.agents = [
        ("executor_agent", object.__new__(ExecutorAgent)),
        ("detect_agent", object.__new__(DetectAgent)),
        ("teacher_agent", object.__new__(TeacherAgent)),
    ]
    phase._last_agent_message = AgentMessage(agent_id="executor_agent")
    phase._teacher_review_pending = True
    calls = []

    async def run_iteration(check=False):
        calls.append(check)
        if check:
            phase._last_agent_message = AgentMessage(agent_id="detect_agent")
        else:
            phase._last_agent_message = AgentMessage(agent_id="teacher_agent")
            phase._teacher_review_pending = False

    with patch.object(phase, "_run_iteration", AsyncMock(side_effect=run_iteration)):
        evaluation = asyncio.run(phase._finish_pending_teacher_review())

    assert calls == [True, False]
    assert evaluation.agent_id == "detect_agent"
    assert phase._last_agent_message.agent_id == "teacher_agent"


def test_terminal_evaluation_is_followed_by_one_teacher_turn(tmp_path):
    phase = make_detect_phase(tmp_path, TeacherMode.STEER.value)
    phase.agents = [
        ("executor_agent", object.__new__(ExecutorAgent)),
        ("detect_agent", object.__new__(DetectAgent)),
        ("teacher_agent", object.__new__(TeacherAgent)),
    ]
    evaluation = AgentMessage(agent_id="detect_agent")
    phase._last_agent_message = evaluation
    phase._teacher_review_pending = True

    async def run_teacher(check=False):
        assert check is False
        phase._last_agent_message = AgentMessage(agent_id="teacher_agent")
        phase._teacher_review_pending = False

    teacher_turn = AsyncMock(side_effect=run_teacher)
    with patch.object(phase, "_run_iteration", teacher_turn):
        result = asyncio.run(phase._finish_pending_teacher_review())

    assert result is evaluation
    assert teacher_turn.await_count == 1


def test_teacher_never_has_environment_access():
    assert TeacherAgent.REQUIRED_RESOURCES == [ResourceType.TEACHER_MODEL]
    assert TeacherAgent.ACCESSIBLE_RESOURCES == [ResourceType.TEACHER_MODEL]
    assert ResourceType.KALI_ENV not in TeacherAgent.OPTIONAL_RESOURCES


def test_any_openrouter_teacher_slug_uses_openrouter_provider():
    model = get_model_info("openrouter/example/custom-teacher", helm=False)
    assert model.provider is ServiceProvider.OPENROUTER
