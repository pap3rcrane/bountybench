import pytest

from messages.trajectory_sanitizer import sanitize_trajectory
from prompts.teacher_response_context import format_teacher_output_for_student


def test_teacher_prompt_transport_is_preserved():
    recorded = {
        "input": "AVAILABLE TRACE",
        "system_prompt": "JUDGE PROMPT",
        "teacher": {"teacher_system_prompt_placement": "system"},
    }

    assert sanitize_trajectory(recorded) == recorded


def test_gemini_reasoning_output_is_preserved():
    recorded = {
        "additional_metadata": {
            "reasoning_output": {
                "type": "gemini_thought_summary",
                "available": True,
                "text": "Check the latest action against the evidence.",
                "parts": ["Check the latest action against the evidence."],
            }
        }
    }

    assert sanitize_trajectory(recorded) == recorded


def test_complete_gemini_api_response_is_preserved_including_usage_tokens():
    api_response = {
        "type": "google.genai.types.GenerateContentResponse",
        "serialization": (
            "pydantic.model_dump(mode=json, by_alias=false, exclude_none=false)"
        ),
        "available": True,
        "data": {
            "response_id": "response-123",
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 20,
                "thoughts_token_count": 5,
                "total_token_count": 35,
            },
        },
    }
    trajectory = {
        "additional_metadata": {
            "gemini_api_response": api_response,
            "input_tokens": 10,
        }
    }

    sanitized = sanitize_trajectory(trajectory)

    assert sanitized == {"additional_metadata": {"gemini_api_response": api_response}}


def test_usage_and_token_metadata_are_removed_recursively():
    result = sanitize_trajectory(
        {
            "workflow_usage": {"total_input_tokens": 4},
            "resources_used": {"model": "teacher"},
            "phase_messages": [
                {
                    "phase_usage": {"input_token": 4},
                    "message": "The word token in prose remains.",
                    "additional_metadata": {
                        "input_tokens": 4,
                        "budget_tokens": 8,
                        "time_taken_in_ms": 10,
                    },
                }
            ],
        }
    )

    assert result == {
        "phase_messages": [
            {
                "message": "The word token in prose remains.",
                "additional_metadata": {"time_taken_in_ms": 10},
            }
        ]
    }


def _feedback_trajectory(
    response,
    *,
    visible,
    mode="observe",
    placement="prepend",
    prompt_file="prompts/system_prompts/single_error_correction_one_alternative.txt",
):
    teacher_message = (
        format_teacher_output_for_student(mode, prompt_file, response)
        if visible
        else ""
    )
    student_input = "MEMORY\n1) [executor_agent] action\n2) [evaluator] result"
    if visible:
        student_input += f"\n3) [teacher_agent] {teacher_message}"
    return {
        "phase_messages": [
            {
                "agent_messages": [
                    {"agent_id": "system", "message": "task"},
                    {"agent_id": "executor_agent", "message": ""},
                    {"agent_id": "evaluator", "message": "result"},
                    {
                        "agent_id": "teacher_agent",
                        "message": teacher_message,
                        "action_messages": [
                            {
                                "message": response,
                                "additional_metadata": {
                                    "teacher": {
                                        "mode": mode,
                                        "teacher_system_prompt_placement": placement,
                                        "teacher_system_prompt_file": prompt_file,
                                    }
                                },
                            }
                        ],
                    },
                    {
                        "agent_id": "executor_agent",
                        "message": "",
                        "action_messages": [
                            {"additional_metadata": {"input": student_input}}
                        ],
                    },
                ]
            }
        ]
    }


@pytest.mark.parametrize(
    "response",
    [
        "PASS\n<END>",
        '{\n  "score": 4,\n  "reason": "The command advanced the task."\n}\n<END>',
        "Prefer candidate 2.\nIt tests the narrowest hypothesis first.\n<END>",
    ],
)
def test_downloaded_observe_feedback_matches_visible_steer_shape(response):
    invisible = _feedback_trajectory(response, visible=False)
    visible = _feedback_trajectory(response, visible=True)

    assert sanitize_trajectory(
        invisible, expose_observe_feedback=True
    ) == visible


def test_normal_logging_does_not_rewrite_observe_history():
    invisible = _feedback_trajectory("PASS\n<END>", visible=False)

    assert sanitize_trajectory(invisible) == invisible


def test_user_placement_observe_feedback_is_reconstructed_when_missing():
    invisible = _feedback_trajectory(
        "Use the narrower command.\n<END>",
        visible=False,
        placement="prepend",
    )
    visible = _feedback_trajectory(
        "Use the narrower command.\n<END>",
        visible=True,
        placement="prepend",
    )

    assert sanitize_trajectory(
        invisible, expose_observe_feedback=True
    ) == visible


def test_already_visible_observe_feedback_is_not_reconstructed():
    visible = _feedback_trajectory("PASS\n<END>", visible=True)

    assert sanitize_trajectory(
        visible, expose_observe_feedback=True
    ) == visible


@pytest.mark.parametrize("mode", ["steer", "objective_rewrite"])
def test_other_teacher_modes_are_not_reconstructed(mode):
    invisible = _feedback_trajectory(
        "Teacher output\n<END>", visible=False, mode=mode
    )

    assert sanitize_trajectory(
        invisible, expose_observe_feedback=True
    ) == invisible


def test_observe_feedback_without_a_following_student_input_is_not_reconstructed():
    incomplete = _feedback_trajectory("PASS\n<END>", visible=False)
    incomplete["phase_messages"][0]["agent_messages"].pop()

    assert sanitize_trajectory(
        incomplete, expose_observe_feedback=True
    ) == incomplete
