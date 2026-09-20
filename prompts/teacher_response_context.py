"""Student-facing context attached to online teacher responses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PromptPath = Optional[Union[str, Path]]


TEACHER_STUDENT_CONTEXT = {
    "observe": {
        "single_error_correction_one_alternative": (
            "The teacher is offering feedback on your latest step. "
            "It identifies a genuine mistake, if any, and provides one corrected "
            "replacement with an explanation. "
            "Use it as additional context to inform your reasoning and next action. "
            "If the teacher returns PASS, it has no input."
        ),
        "single_error_correction_multiple_alternatives": (
            "The teacher is offering feedback on your latest step. "
            "It identifies a genuine mistake, if any, and presents three distinct "
            "corrective approaches with their tradeoffs. "
            "Use it as additional context to inform your reasoning and next action. "
            "If the teacher returns PASS, it has no input."
        ),
        "optimizer": (
            "The teacher is offering feedback on your latest step. "
            "It provides a clearer and more concise version of your reasoning without "
            "changing its substance, conclusion, or action. "
            "Use it as additional context to inform your reasoning and next action."
        ),
        "principle_extraction": (
            "The teacher is offering feedback on your latest step. "
            "It extracts a transferable principle from the step and explains when that "
            "principle may be useful. "
            "Use it as additional context to inform your reasoning and next action."
        ),
        "student_nudging": (
            "The teacher is offering feedback on your latest step. "
            "It identifies a meaningful issue, if any, and provides progressively "
            "stronger hints without directly giving you the correction. "
            "Use it as additional context to inform your reasoning and next action. "
            "If the teacher returns PASS, it has no input."
        ),
    },
    "steer": {
        "outcome_evaluation": (
            "The teacher is offering an evaluation of your latest step. "
            "It scores how much the step advanced the objective and cites trajectory "
            "evidence supporting that score. "
            "Use it as additional context to inform your reasoning and next action."
        ),
        "teacher_answer_comparison": (
            "The teacher is offering an evaluation of your latest step. "
            "It compares the step with a concrete hypothetical alternative and scores "
            "the step relative to that benchmark. "
            "Use it as additional context to inform your reasoning and next action."
        ),
        "comparative_ranking": (
            "The teacher is offering an evaluation of candidate actions for your next "
            "step. It ranks the candidates and explains their key differences and "
            "likely outcomes. Use it as additional context to inform your reasoning "
            "and next action."
        ),
        "reasoning_quality_fidelity": (
            "The teacher is offering an evaluation of your latest step. "
            "It separately assesses whether your reasoning is sound and whether your "
            "action faithfully follows that reasoning. "
            "Use it as additional context to inform your reasoning and next action."
        ),
        "rubric_based_evaluation": (
            "The teacher is offering an evaluation of your latest step. "
            "It scores the step against an eight-criterion rubric and provides evidence "
            "for each score. Use it as additional context to inform your reasoning and "
            "next action."
        ),
    },
}


BASELINE_TEACHER_STUDENT_CONTEXT = {
    "observe": "",
    "steer": "",
}


GENERIC_TEACHER_STUDENT_CONTEXT = {
    "observe": (
        "The teacher is offering feedback on your latest step. "
        "It provides guidance derived from the task and trajectory. "
        "Use it as additional context to inform your reasoning and next action."
    ),
    "steer": (
        "The teacher is offering an evaluation of your latest step. "
        "It provides an assessment derived from the task and trajectory. "
        "Use it as additional context to inform your reasoning and next action."
    ),
}


def teacher_student_context(mode: str, system_prompt_file: PromptPath) -> str:
    """Return context for a prompt-specific or no-custom-prompt teacher output."""
    if mode not in TEACHER_STUDENT_CONTEXT:
        raise ValueError(f"Unsupported online teacher mode: {mode}")
    if system_prompt_file is None:
        return BASELINE_TEACHER_STUDENT_CONTEXT[mode]

    prompt_name = Path(system_prompt_file).stem
    return TEACHER_STUDENT_CONTEXT[mode].get(
        prompt_name, GENERIC_TEACHER_STUDENT_CONTEXT[mode]
    )


def format_teacher_output_for_student(
    mode: str, system_prompt_file: PromptPath, response: str
) -> str:
    """Attach a concise explanation before raw teacher output shown to a student."""
    context = teacher_student_context(mode, system_prompt_file)
    if not context:
        return response
    return f"{context}\n\n{response}"
