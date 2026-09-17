import json
from pathlib import Path

from scripts.prepare_matrix_retry import build_exclusions


def _configuration(status, teacher_type, log_path, run_number):
    return {
        "record_type": "configuration",
        "status": status,
        "teacher_type": teacher_type,
        "repo_name": "repo",
        "bounty_number": "0",
        "workflow_type": "detect_workflow",
        "system_prompt_name": "prompt",
        "system_prompt_placement": "system",
        "run_number": run_number,
        "configuration_log_path": str(log_path),
    }


def test_retry_manifest_excludes_skips_and_completed_teacher_rewrites(tmp_path):
    completed_log = tmp_path / "completed.log"
    completed_log.write_text(
        'BOUNTYBENCH_EVENT {"event": "teacher_rewrite_finished", '
        '"status": "success"}\n'
    )
    failed_log = tmp_path / "failed.log"
    failed_log.write_text(
        'BOUNTYBENCH_EVENT {"event": "teacher_rewrite_finished", '
        '"status": "failure"}\n'
    )
    status_path = tmp_path / "run_status.jsonl"
    records = [
        _configuration("success", "objective_rewrite", completed_log, 1),
        _configuration("skipped", "objective_rewrite", failed_log, 2),
        _configuration("failure", "objective_rewrite", completed_log, 3),
        _configuration("failure", "objective_rewrite", failed_log, 4),
        _configuration("failure", "observe", completed_log, 5),
    ]
    status_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    exclusions, counts = build_exclusions(status_path)

    assert [record["run_number"] for record in exclusions] == [2, 3]
    assert counts == {
        "successful": 1,
        "failed_to_retry": 2,
        "already_skipped": 1,
        "completed_objective_rewrites": 1,
    }
