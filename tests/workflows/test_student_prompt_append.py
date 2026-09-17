from pathlib import Path

from workflows.base_workflow import BaseWorkflow


COMPARATIVE_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "prompts/student_prompts/comparative_ranking_candidates.txt"
)


def test_student_prompt_file_is_appended_after_rendered_benchmark_task(tmp_path):
    prompt_file = tmp_path / "student_append.txt"
    prompt_file.write_text("Generate three candidates, then select one.\n")

    rendered = BaseWorkflow._append_student_prompt_file(
        "ORIGINAL RENDERED BENCHMARK TASK\n", str(prompt_file)
    )

    assert rendered == (
        "ORIGINAL RENDERED BENCHMARK TASK\n\n"
        "Generate three candidates, then select one.\n"
    )


def test_student_prompt_append_rejects_empty_file(tmp_path):
    prompt_file = tmp_path / "empty.txt"
    prompt_file.write_text("\n")

    try:
        BaseWorkflow._append_student_prompt_file("task", str(prompt_file))
    except ValueError as error:
        assert "is empty" in str(error)
    else:
        raise AssertionError("Expected an empty student prompt file to be rejected")


def test_comparative_ranking_student_prompt_has_requested_wording():
    prompt = COMPARATIVE_PROMPT.read_text()

    assert "candidate_A" in prompt
    assert "candidate_B" in prompt
    assert "candidate_C" in prompt
    assert "selected_candidate" in prompt
    assert "exactly one concrete executable command" in prompt
    assert "teacher" not in prompt.lower()
    assert "thought field" not in prompt.lower()
    assert "COMPARATIVE-RANKING CANDIDATE REQUIREMENT" not in prompt
