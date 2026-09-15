import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "print_teacher_prompts.py"


def _run_script(*args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_prints_complete_system_placement_payload():
    output = _run_script(
        "--workflow",
        "exploit_workflow",
        "--prompt",
        "outcome_evaluation",
        "--placement",
        "system",
        "--format",
        "json",
    )

    assert output["request_count"] == 1
    request = output["requests"][0]
    prompt = (
        (PROJECT_ROOT / "prompts/system_prompts/outcome_evaluation.txt")
        .read_text()
        .strip()
    )
    assert request["teacher_mode"] == "steer"
    system_instruction = request["gemini_api"]["GenerativeModel"][
        "system_instruction"
    ]
    assert system_instruction.startswith(prompt + "\n\n")
    assert "ORIGINAL BENCHMARK TASK:" in system_instruction
    assert "attempting to exploit" in system_instruction
    assert prompt not in request["gemini_api"]["generate_content"]["contents"]
    assert "ORIGINAL BENCHMARK TASK:" not in (
        request["gemini_api"]["generate_content"]["contents"]
    )


def test_prints_complete_prepend_payload():
    output = _run_script(
        "--workflow",
        "detect_workflow",
        "--prompt",
        "optimizer",
        "--placement",
        "prepend",
        "--format",
        "json",
    )

    request = output["requests"][0]
    prompt = (PROJECT_ROOT / "prompts/system_prompts/optimizer.txt").read_text().strip()
    assert request["teacher_mode"] == "observe"
    assert request["gemini_api"]["GenerativeModel"]["system_instruction"] is None
    assert request["gemini_api"]["generate_content"]["contents"].startswith(
        prompt + "\n\n"
    )
    assert (
        "agent=executor_agent" in request["gemini_api"]["generate_content"]["contents"]
    )
    assert (
        "agent=detect_agent"
        not in request["gemini_api"]["generate_content"]["contents"]
    )


def test_prints_objective_rewrite_payload_with_three_source_runs():
    output = _run_script(
        "--workflow",
        "patch_workflow",
        "--prompt",
        "oversight_task",
        "--placement",
        "system",
        "--format",
        "json",
    )

    request = output["requests"][0]
    contents = request["gemini_api"]["generate_content"]["contents"]
    system_instruction = request["gemini_api"]["GenerativeModel"][
        "system_instruction"
    ]
    assert request["teacher_mode"] == "objective_rewrite"
    assert system_instruction.count("ORIGINAL BENCHMARK TASK:") == 3
    assert "ORIGINAL BENCHMARK TASK:" not in contents
    assert contents.count("AVAILABLE TRACE (oldest to newest):") == 3
    assert "attempting to patch" in system_instruction


def test_default_matrix_contains_every_workflow_prompt_placement_pair(tmp_path):
    output_path = tmp_path / "teacher-prompts.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(output_path.read_text())
    assert output["request_count"] == 3 * 15 * 2
    assert "Wrote 90 complete Gemini prompt payloads" in result.stdout
