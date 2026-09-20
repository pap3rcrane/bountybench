import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.download_vm_matrix_traces import (
    OBJECTIVE_INDEX_FILE,
    OUTPUT_LAYOUT,
    SELECTED_ENVIRONMENTS,
    build_objective_index,
    collect_attempts,
    discard_unstable_files,
    merge_vm_attempts,
)


def write_export(
    root: Path,
    *,
    teacher_type: str = "observe",
    repository: str = "astropy",
    workflow_type: str = "detect_workflow",
    run_number: int = 2,
    roles: tuple[str, ...] = ("normal",),
    configuration_status: str = "success",
    failed_roles: tuple[str, ...] = (),
    attempt_name: str = "attempt",
    finished_at: str = "2026-09-18T12:00:00+00:00",
    trajectory_marker: str = "default",
) -> None:
    status_path = root / "runs" / teacher_type / attempt_name / "run_status.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    configuration_id = f"{repository}-{workflow_type}-{run_number}-{attempt_name}"
    records = [
        {
            "record_type": "batch_start",
            "batch_id": f"batch-{configuration_id}",
            "teacher_type": teacher_type,
            "teacher_model": "google/gemini-test",
        }
    ]
    common = {
        "configuration_id": configuration_id,
        "repo_name": repository,
        "bounty_number": "0",
        "workflow_type": workflow_type,
        "teacher_type": teacher_type,
        "system_prompt_name": "optimizer",
        "system_prompt_placement": "system",
        "run_number": run_number,
        "finished_at": finished_at,
    }
    configuration_log_path = (
        root / "batch_logs" / teacher_type / attempt_name / f"{configuration_id}.log"
    )
    for role in roles:
        trajectory_path = root / "logs" / f"{configuration_id}-{role}.json"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(
            json.dumps(
                {
                    "role": role,
                    "attempt_marker": trajectory_marker,
                    "workflow_usage": {"total_input_tokens": 12},
                    "resources_used": {"model": {"max_output_tokens": 100}},
                    "phase_messages": [
                        {
                            "phase_usage": {"input_token": 12},
                            "agent_messages": [
                                {
                                    "additional_metadata": {
                                        "input_tokens": 12,
                                        "max_output_tokens": 100,
                                        "time_taken_in_ms": 25,
                                        "input": "AVAILABLE TRACE",
                                        "system_prompt": "JUDGE PROMPT",
                                        "teacher": {
                                            "teacher_system_prompt_placement": "system"
                                        },
                                    },
                                    "message": "The word token in prose is preserved.",
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "record_type": "student_run",
                **common,
                "run_role": role,
                "status": "failure" if role in failed_roles else "success",
                "workflow_log_path": str(trajectory_path),
                "configuration_log_path": str(configuration_log_path),
            }
        )
    records.append(
        {
            "record_type": "configuration",
            **common,
            "status": configuration_status,
        }
    )
    status_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    if teacher_type == "objective_rewrite":
        objective_path = (
            root
            / "batch_logs"
            / "matrix_workers"
            / attempt_name
            / "backend-worker-1"
            / "generated_objectives"
            / f"{configuration_id}_objective.json"
        )
        objective_path.parent.mkdir(parents=True, exist_ok=True)
        objective_path.write_text(
            json.dumps({"objective": f"objective-{trajectory_marker}"}),
            encoding="utf-8",
        )
        teacher_trace_path = objective_path.with_name(
            objective_path.name.replace("_objective.json", "_teacher_trace.json")
        )
        teacher_trace_path.write_text(
            json.dumps(
                {
                    "teacher_mode": "objective_rewrite",
                    "system_prompt": "TASK DESIGNER PROMPT",
                    "input": "THREE SOURCE TRACES",
                    "raw_response": "raw teacher response",
                    "reasoning_output": {
                        "type": "gemini_thought_summary",
                        "available": True,
                        "text": "Compare the three source traces.",
                        "parts": ["Compare the three source traces."],
                    },
                    "response": "teacher response",
                    "objective": f"objective-{trajectory_marker}",
                }
            ),
            encoding="utf-8",
        )
        (root / OBJECTIVE_INDEX_FILE).write_text(
            json.dumps(
                {
                    str(configuration_log_path.relative_to(root)): {
                        "objective_path": str(objective_path.relative_to(root)),
                        "teacher_trace_path": str(
                            teacher_trace_path.relative_to(root)
                        ),
                    }
                }
            ),
            encoding="utf-8",
        )


def test_normal_run_is_one_flat_trajectory_and_removes_legacy_folder(tmp_path):
    staging = tmp_path / "export"
    destination = tmp_path / "download"
    write_export(staging)

    configuration_dir = (
        destination
        / "astropy_bounty_0_detect_workflow"
        / "astropy_bounty_0_detect_workflow__system__critiquer__optimizer"
    )
    legacy_run_dir = configuration_dir / "run2"
    legacy_run_dir.mkdir(parents=True)
    (legacy_run_dir / "trajectory.json").write_text("old", encoding="utf-8")
    (legacy_run_dir / "full.log").write_text("old", encoding="utf-8")
    (legacy_run_dir / "batch.log").write_text("old", encoding="utf-8")

    attempt = collect_attempts(staging, "test-vm")[0]
    result = merge_vm_attempts([attempt], destination, "test-vm")

    assert sorted(path.name for path in configuration_dir.iterdir()) == ["run2.json"]
    trajectory = json.loads((configuration_dir / "run2.json").read_text())
    assert trajectory == {
        "role": "normal",
        "attempt_marker": "default",
        "phase_messages": [
            {
                "agent_messages": [
                    {
                        "additional_metadata": {
                            "time_taken_in_ms": 25,
                            "input": "AVAILABLE TRACE",
                            "system_prompt": "JUDGE PROMPT",
                            "teacher": {"teacher_system_prompt_placement": "system"},
                        },
                        "message": "The word token in prose is preserved.",
                    }
                ]
            }
        ],
    }
    assert result["trajectory_runs_written"] == 1
    assert result["records"][0]["layout"] == OUTPUT_LAYOUT
    assert set(result["records"][0]["files"]) == {"trajectory"}


def test_task_designer_preserves_sources_and_sanitized_teacher_trace(tmp_path):
    staging = tmp_path / "export"
    destination = tmp_path / "download"
    write_export(
        staging,
        teacher_type="objective_rewrite",
        repository="bentoml",
        roles=("source_1", "source_2", "source_3"),
        configuration_status="failure",
        run_number=1,
    )

    attempt = collect_attempts(staging, "test-vm")[0]
    result = merge_vm_attempts([attempt], destination, "test-vm")

    configuration_dir = (
        destination
        / "bentoml_bounty_0_detect_workflow"
        / "bentoml_bounty_0_detect_workflow__system__task_designer__optimizer"
    )
    assert sorted(path.name for path in configuration_dir.iterdir()) == [
        "run1_source_1.json",
        "run1_source_2.json",
        "run1_source_3.json",
        "run1_teacher_trace.json",
    ]
    teacher_trace = json.loads(
        (configuration_dir / "run1_teacher_trace.json").read_text()
    )
    assert teacher_trace["input"] == "THREE SOURCE TRACES"
    assert teacher_trace["raw_response"] == "raw teacher response"
    assert teacher_trace["reasoning_output"] == {
        "type": "gemini_thought_summary",
        "available": True,
        "text": "Compare the three source traces.",
        "parts": ["Compare the three source traces."],
    }
    assert "response" not in teacher_trace
    assert "objective" not in teacher_trace
    assert "record_source" not in teacher_trace
    assert teacher_trace["model"] == "google/gemini-test"
    assert teacher_trace["system_prompt_placement"] == "system"
    assert teacher_trace["configuration"]["run_number"] == "1"
    assert result["trajectory_runs_written"] == 1
    assert result["records"][0]["configuration_status"] == "failure"
    assert set(result["records"][0]["files"]) == {
        "source_1_trajectory",
        "source_2_trajectory",
        "source_3_trajectory",
        "teacher_trace",
    }


def test_historical_task_designer_log_reconstructs_complete_teacher_trace(tmp_path):
    staging = tmp_path / "export"
    objective = (
        staging
        / "batch_logs/matrix_workers/launch/backend-worker-1/generated_objectives"
        / "repo_bounty_0_detect_workflow_2026-09-17_01-02-03_objective.json"
    )
    objective.parent.mkdir(parents=True)
    objective.write_text(json.dumps({"objective": "clean objective"}))
    log = staging / "batch_logs/objective_rewrite/batch/configuration.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                "2026-09-17 INFO [resources/model_resource/model_resource.py:1]",
                "Model system prompt:",
                "TASK DESIGNER SYSTEM PROMPT",
                "2026-09-17 INFO [resources/model_resource/model_resource.py:2]",
                "Model input (truncated if over max tokens):",
                "SOURCE RUN 1",
                "SOURCE RUN 2",
                "SOURCE RUN 3",
                "2026-09-17 INFO [resources/model_resource/model_resource.py:3]",
                "Unparsed LM Response:",
                "content:",
                "raw teacher response",
                "",
                "input_tokens:",
                "123",
                "output_tokens:",
                "45",
                "2026-09-17 INFO [workflows/teacher_modes.py:4]",
                "TEACHER RESPONSE (objective_rewrite)",
                "processed teacher response",
                "<END>",
                "================================================================================",
                "Wrote rewritten objective: ",
                "generated_objectives/repo_bounty_0_detect_workflow_2026-09-17_01-",
                "02-03_objective.json",
            ]
        )
    )

    build_objective_index(staging)

    index = json.loads((staging / OBJECTIVE_INDEX_FILE).read_text())
    entry = index[str(log.relative_to(staging))]
    trace = json.loads((staging / entry["teacher_trace_path"]).read_text())
    assert trace["system_prompt"] == "TASK DESIGNER SYSTEM PROMPT"
    assert trace["input"] == "SOURCE RUN 1\nSOURCE RUN 2\nSOURCE RUN 3"
    assert trace["raw_response"] == "raw teacher response"
    assert "response" not in trace
    assert "objective" not in trace
    assert "record_source" not in trace
    assert "input_tokens" not in json.dumps(trace)


def test_failed_task_designer_parse_with_raw_output_is_downloaded(tmp_path):
    staging = tmp_path / "export"
    destination = tmp_path / "download"
    write_export(
        staging,
        teacher_type="objective_rewrite",
        repository="bentoml",
        roles=("source_1", "source_2", "source_3"),
        configuration_status="failure",
        run_number=4,
    )
    (staging / OBJECTIVE_INDEX_FILE).unlink()
    for artifact in staging.glob("**/generated_objectives/*.json"):
        artifact.unlink()
    configuration_log = (
        staging
        / "batch_logs/objective_rewrite/attempt"
        / "bentoml-detect_workflow-4-attempt.log"
    )
    configuration_log.parent.mkdir(parents=True, exist_ok=True)
    configuration_log.write_text(
        "\n".join(
            [
                "2026-09-17 INFO [resources/model_resource/model_resource.py:1]",
                "Model system prompt:",
                "TASK DESIGNER SYSTEM PROMPT",
                "2026-09-17 INFO [resources/model_resource/model_resource.py:2]",
                "Model input (truncated if over max tokens):",
                "THREE SOURCE TRACES",
                "2026-09-17 INFO [resources/model_resource/model_resource.py:3]",
                "Unparsed LM Response:",
                "content:",
                "```json",
                '{"objective": "useful but fenced"}',
                "```",
                "",
                "input_tokens:",
                "123",
                "output_tokens:",
                "45",
                "2026-09-17 INFO [workflows/teacher_modes.py:4]",
                "TEACHER RESPONSE (objective_rewrite)",
                "```json",
                '{"objective": "useful but fenced"}',
                "```",
                "2026-09-17 ERROR [workflows/teacher_modes.py:5]",
                "Expecting value: line 1 column 1 (char 0)",
            ]
        ),
        encoding="utf-8",
    )

    result = merge_vm_attempts(
        collect_attempts(staging, "test-vm"), destination, "test-vm"
    )

    record = result["records"][0]
    assert result["trajectory_runs_written"] == 1
    assert set(record["files"]) == {
        "source_1_trajectory",
        "source_2_trajectory",
        "source_3_trajectory",
        "teacher_trace",
    }
    trace = json.loads(Path(record["files"]["teacher_trace"]).read_text())
    assert trace["raw_response"] == (
        "```json\n{\"objective\": \"useful but fenced\"}\n```"
    )
    assert trace["input"] == "THREE SOURCE TRACES"


def test_incomplete_task_designer_sources_are_not_downloaded(tmp_path):
    staging = tmp_path / "export"
    destination = tmp_path / "download"
    write_export(
        staging,
        teacher_type="objective_rewrite",
        repository="bentoml",
        roles=("source_1", "source_2", "source_3"),
        configuration_status="failure",
        failed_roles=("source_3",),
    )

    attempt = collect_attempts(staging, "test-vm")[0]
    result = merge_vm_attempts([attempt], destination, "test-vm")

    assert result["trajectory_runs_written"] == 0
    assert result["records"] == []


def test_newest_complete_attempt_wins_across_status_directories_and_vms(tmp_path):
    destination = tmp_path / "download"
    first_vm = tmp_path / "vm-a"
    write_export(
        first_vm,
        attempt_name="archive/older",
        finished_at="2026-09-18T10:00:00+00:00",
        trajectory_marker="older",
    )
    write_export(
        first_vm,
        attempt_name="active/newest-incomplete",
        finished_at="2026-09-18T14:00:00+00:00",
        trajectory_marker="newest-incomplete",
        failed_roles=("normal",),
    )
    write_export(
        first_vm,
        attempt_name="retry/newest-complete-on-vm-a",
        finished_at="2026-09-18T12:00:00+00:00",
        trajectory_marker="newest-complete-on-vm-a",
    )

    first_attempts = collect_attempts(first_vm, "vm-a")
    assert len(first_attempts) == 3
    first_result = merge_vm_attempts(first_attempts, destination, "vm-a")
    selected_path = Path(first_result["records"][0]["files"]["trajectory"])
    assert json.loads(selected_path.read_text())["attempt_marker"] == (
        "newest-complete-on-vm-a"
    )

    second_vm = tmp_path / "vm-b"
    write_export(
        second_vm,
        attempt_name="later-vm",
        finished_at="2026-09-18T13:00:00+00:00",
        trajectory_marker="newest-across-vms",
    )
    second_result = merge_vm_attempts(
        collect_attempts(second_vm, "vm-b"), destination, "vm-b"
    )
    assert second_result["records"][0]["source_vm"] == "vm-b"
    assert json.loads(selected_path.read_text())["attempt_marker"] == (
        "newest-across-vms"
    )

    third_vm = tmp_path / "vm-c"
    write_export(
        third_vm,
        attempt_name="stale-vm",
        finished_at="2026-09-18T11:00:00+00:00",
        trajectory_marker="stale",
    )
    final_result = merge_vm_attempts(
        collect_attempts(third_vm, "vm-c"), destination, "vm-c"
    )
    assert final_result["updates_saved_from_last_vm"] == 0
    assert final_result["records"][0]["source_vm"] == "vm-b"
    assert json.loads(selected_path.read_text())["attempt_marker"] == (
        "newest-across-vms"
    )


def test_malformed_newest_judge_uses_sanitized_older_backup(tmp_path):
    staging = tmp_path / "vm"
    destination = tmp_path / "download"
    write_export(
        staging,
        teacher_type="steer",
        repository="mlflow",
        workflow_type="exploit_workflow",
        attempt_name="older",
        finished_at="2026-09-18T10:00:00+00:00",
        trajectory_marker="older",
    )
    write_export(
        staging,
        teacher_type="steer",
        repository="mlflow",
        workflow_type="exploit_workflow",
        attempt_name="newer",
        finished_at="2026-09-18T11:00:00+00:00",
        trajectory_marker="newer",
    )
    newest = staging / "logs" / "mlflow-exploit_workflow-2-newer-normal.json"
    newest.write_text("{unfinished", encoding="utf-8")

    result = merge_vm_attempts(collect_attempts(staging, "vm"), destination, "vm")

    record = result["records"][0]
    trajectory = json.loads(Path(record["files"]["trajectory"]).read_text())
    assert trajectory["attempt_marker"] == "older"
    assert "workflow_usage" not in trajectory
    assert "input_tokens" not in json.dumps(trajectory)
    assert record["used_older_backup"] is True
    assert result["older_backups_used_from_last_vm"] == 1


def test_changed_newest_file_does_not_replace_existing_older_download(tmp_path):
    destination = tmp_path / "download"
    older = tmp_path / "older"
    write_export(
        older,
        attempt_name="older",
        finished_at="2026-09-18T10:00:00+00:00",
        trajectory_marker="older",
    )
    first = merge_vm_attempts(collect_attempts(older, "vm-a"), destination, "vm-a")
    trajectory_path = Path(first["records"][0]["files"]["trajectory"])

    newer = tmp_path / "newer"
    write_export(
        newer,
        attempt_name="newer",
        finished_at="2026-09-18T11:00:00+00:00",
        trajectory_marker="newer",
    )
    changed_path = "logs/astropy-detect_workflow-2-newer-normal.json"
    unstable = discard_unstable_files(
        staging_root=newer,
        before={changed_path: ("100", "1.0")},
        after={changed_path: ("101", "2.0")},
    )
    second = merge_vm_attempts(
        collect_attempts(newer, "vm-b"),
        destination,
        "vm-b",
        unstable,
    )

    assert unstable == {changed_path}
    assert json.loads(trajectory_path.read_text())["attempt_marker"] == "older"
    assert second["updates_saved_from_last_vm"] == 0

def test_selected_environment_groups_include_only_the_25_requested():
    roles = Counter(role for role, _repo, _bounty, _workflow in SELECTED_ENVIRONMENTS)
    workflows: dict[str, Counter[str]] = defaultdict(Counter)
    for role, _repository, _bounty, workflow in SELECTED_ENVIRONMENTS:
        workflows[role][workflow] += 1

    assert roles == {"observe": 8, "steer": 8, "objective_rewrite": 9}
    assert workflows["observe"] == {
        "exploit_workflow": 2,
        "detect_workflow": 3,
        "patch_workflow": 3,
    }
    assert workflows["steer"] == {
        "exploit_workflow": 3,
        "detect_workflow": 3,
        "patch_workflow": 2,
    }
    assert workflows["objective_rewrite"] == {
        "exploit_workflow": 3,
        "detect_workflow": 3,
        "patch_workflow": 3,
    }
    assert not any(
        repository == "InvokeAI" for _, repository, _, _ in SELECTED_ENVIRONMENTS
    )
    assert not any(
        repository == "llama_index" for _, repository, _, _ in SELECTED_ENVIRONMENTS
    )
    assert any(repository == "node" for _, repository, _, _ in SELECTED_ENVIRONMENTS)


def test_excluded_environment_is_removed_from_existing_download(tmp_path):
    destination = tmp_path / "download"
    configuration_dir = (
        destination
        / "llama_index_bounty_0_patch_workflow"
        / "llama_index_bounty_0_patch_workflow__system__judge_verifier__optimizer"
    )
    configuration_dir.mkdir(parents=True)
    trajectory_path = configuration_dir / "run1.json"
    trajectory_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "records": [
            {
                "logical_key": [
                    "llama_index",
                    "0",
                    "patch_workflow",
                    "steer",
                    "optimizer",
                    "system",
                    "1",
                ],
                "effective_success": True,
                "files": {"trajectory": str(trajectory_path)},
            }
        ]
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    result = merge_vm_attempts([], destination, "test-vm")

    assert result["records"] == []
    assert result["excluded_existing_records_removed"] == 1
    assert not trajectory_path.exists()
