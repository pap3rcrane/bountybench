import json
import threading
import time
from argparse import Namespace
from pathlib import Path

from scripts.run_teacher_prompt_matrix import (
    Environment,
    MatrixRunner,
    build_configurations,
    load_successful_configuration_keys,
    parse_runner_event,
)

PROMPT_FILE = "prompts/system_prompts/optimizer.txt"


def _args(*, mode):
    return Namespace(
        matrix_name=f"test_{mode}",
        teacher_mode=mode,
        prompt_placement="both",
        skip_configurations_from=[],
        prompt_file=[PROMPT_FILE],
        environment=[Environment("lunary", "0", "detect_workflow")],
        launcher="run_teacher_matrix.sh",
        launcher_argument=["--dry-run"],
        dry_run=False,
        prune_dind_between_repositories=False,
        verbose=False,
        no_progress=True,
        jobs=1,
    )


def _event(event, **fields):
    return "BOUNTYBENCH_EVENT " + json.dumps(
        {"event": event, "timestamp": "2026-09-14T12:00:00-04:00", **fields}
    )


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_launcher_shape_contains_495_configurations():
    environments = [
        Environment(f"repo-{number}", "0", "detect_workflow") for number in range(9)
    ]
    prompts = [f"prompts/system_prompts/prompt-{number}.txt" for number in range(5)]

    configurations = build_configurations(
        environments=environments,
        prompt_files=prompts,
        repetitions=5,
    )

    assert len(configurations) == 495
    assert len({item.configuration_id for item in configurations}) == 495
    assert sum(item.placement == "none" for item in configurations) == 45


def test_user_prompt_placement_runs_only_prepend_and_baseline():
    configurations = build_configurations(
        environments=[Environment("lunary", "0", "detect_workflow")],
        prompt_files=[PROMPT_FILE],
        repetitions=5,
        placements=("prepend",),
    )

    assert len(configurations) == 10
    assert {item.placement for item in configurations} == {"prepend", "none"}


def test_runner_event_parser_ignores_normal_output():
    assert parse_runner_event("Finished iteration 1") is None
    assert parse_runner_event(_event("student_run_started", run_role="normal")) == {
        "event": "student_run_started",
        "timestamp": "2026-09-14T12:00:00-04:00",
        "run_role": "normal",
    }


def test_skip_source_uses_only_final_successful_configurations(tmp_path):
    status_path = tmp_path / "run_status.jsonl"
    records = [
        {
            "record_type": "configuration",
            "configuration_id": "completed-success",
            "status": "success",
            "teacher_type": "observe",
            "repo_name": "lunary",
            "bounty_number": "0",
            "workflow_type": "detect_workflow",
            "system_prompt_name": "optimizer",
            "system_prompt_placement": "prepend",
            "run_number": 1,
        },
        {
            "record_type": "configuration",
            "configuration_id": "completed-failure",
            "status": "failure",
        },
        {
            "record_type": "student_run",
            "configuration_id": "still-in-progress",
            "status": "success",
        },
    ]
    status_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert load_successful_configuration_keys([str(status_path)]) == {
        (
            "observe",
            "lunary",
            "0",
            "detect_workflow",
            "optimizer",
            "prepend",
            1,
        )
    }


def test_skip_matching_does_not_depend_on_configuration_sequence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    status_path = tmp_path / "previous.jsonl"
    status_path.write_text(
        json.dumps(
            {
                "record_type": "configuration",
                "configuration_id": "different-sequence-prefix",
                "status": "success",
                "teacher_type": "observe",
                "repo_name": "lunary",
                "bounty_number": "0",
                "workflow_type": "detect_workflow",
                "system_prompt_name": "optimizer",
                "system_prompt_placement": "system",
                "run_number": 1,
            }
        )
        + "\n"
    )
    args = _args(mode="observe")
    args.prompt_placement = "system"
    args.skip_configurations_from = [str(status_path)]
    runner = MatrixRunner(args)
    try:
        assert runner.skip_configuration_ids == {
            "0001_lunary_bounty_0_detect_workflow_optimizer_system_repeat_1"
        }
    finally:
        runner.progress.close()
        runner.recorder.close()


def test_compact_status_records_successful_student_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    runner = MatrixRunner(_args(mode="observe"))
    monkeypatch.setattr(runner, "_preflight", lambda: None)

    class SuccessfulProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(
                [
                    _event("student_run_started", run_role="normal") + "\n",
                    "Finished iteration 0 of DetectPhase with executor_agent\n",
                    _event(
                        "student_run_finished",
                        run_role="normal",
                        status="success",
                        workflow_log_path="logs/test.json",
                        error=None,
                    )
                    + "\n",
                ]
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        "scripts.run_teacher_prompt_matrix.subprocess.Popen", SuccessfulProcess
    )

    assert runner.run() == 0
    records = _records(runner.status_path)
    configurations = [
        record for record in records if record["record_type"] == "configuration"
    ]
    students = [record for record in records if record["record_type"] == "student_run"]

    assert len(configurations) == 15
    assert len(students) == 15
    assert {record["status"] for record in configurations} == {"success"}
    assert {record["status"] for record in students} == {"success"}
    assert students[0]["workflow_log_path"] == str(
        Path(__file__).resolve().parents[1] / "logs/test.json"
    )
    assert students[0]["repo_name"] == "lunary"
    assert students[0]["bounty_number"] == "0"
    assert students[0]["run_number"] == 1
    assert students[0]["teacher_type"] == "observe"
    assert students[0]["backend_container"] == "backend-service"
    assert students[0]["error"] is None
    assert any(record["system_prompt_name"] == "none" for record in students)


def test_objective_failure_records_later_students_as_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    runner = MatrixRunner(_args(mode="objective_rewrite"))
    monkeypatch.setattr(runner, "_preflight", lambda: None)

    class FailedSourceProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(
                [
                    _event("student_run_started", run_role="source_1") + "\n",
                    _event(
                        "student_run_finished",
                        run_role="source_1",
                        status="success",
                        workflow_log_path="logs/source-1.json",
                        error=None,
                    )
                    + "\n",
                    _event("student_run_started", run_role="source_2") + "\n",
                    _event(
                        "student_run_finished",
                        run_role="source_2",
                        status="failure",
                        workflow_log_path="logs/source-2.json",
                        error="RuntimeError: source failed",
                    )
                    + "\n",
                ]
            )

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(
        "scripts.run_teacher_prompt_matrix.subprocess.Popen", FailedSourceProcess
    )

    assert runner.run() == 1
    records = _records(runner.status_path)
    first_id = runner.configurations[0].configuration_id
    students = [
        record
        for record in records
        if record["record_type"] == "student_run"
        and record["configuration_id"] == first_id
    ]

    assert [record["run_role"] for record in students] == [
        "source_1",
        "source_2",
        "source_3",
        "rewritten_objective",
    ]
    assert [record["status"] for record in students] == [
        "success",
        "failure",
        "skipped",
        "skipped",
    ]
    assert students[1]["error"] == "RuntimeError: source failed"
    assert students[2]["error"] == "RuntimeError: source failed"


def test_prunes_dind_once_after_each_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    args = _args(mode="observe")
    args.environment = [
        Environment("repo-a", "0", "detect_workflow"),
        Environment("repo-a", "1", "patch_workflow"),
        Environment("repo-b", "0", "exploit_workflow"),
    ]
    args.prune_dind_between_repositories = True
    runner = MatrixRunner(args)
    monkeypatch.setattr(runner, "_preflight", lambda: None)
    monkeypatch.setattr(
        runner,
        "_run_configuration",
        lambda configuration, backend_container: "success",
    )
    pruned_repositories = []
    monkeypatch.setattr(
        runner,
        "_prune_dind",
        lambda repository, backend_container: pruned_repositories.append(
            (repository, backend_container)
        ),
    )

    assert runner.run() == 0
    assert pruned_repositories == [
        ("repo-a", "backend-service"),
        ("repo-b", "backend-service"),
    ]


def test_dind_prune_preserves_tagged_images(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    runner = MatrixRunner(_args(mode="observe"))
    commands = []

    class SuccessfulPrune:
        returncode = 0
        stdout = "Total reclaimed space: 1GB\n"
        stderr = ""

    def run_command(command, **kwargs):
        commands.append(command)
        return SuccessfulPrune()

    monkeypatch.setattr("scripts.run_teacher_prompt_matrix.subprocess.run", run_command)
    try:
        runner._prune_dind("lunary", "backend-service")
    finally:
        runner.progress.close()
        runner.recorder.close()

    assert commands == [
        [
            "docker",
            "exec",
            "backend-service",
            "docker",
            "system",
            "prune",
            "-f",
            "--volumes",
        ]
    ]
    assert "-a" not in commands[0]


def test_parallel_jobs_never_overlap_the_same_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    args = _args(mode="observe")
    args.jobs = 2
    args.environment = [
        Environment("repo-a", "0", "detect_workflow"),
        Environment("repo-a", "1", "patch_workflow"),
        Environment("repo-b", "0", "exploit_workflow"),
    ]
    runner = MatrixRunner(args)
    monkeypatch.setattr(runner, "_preflight", lambda: None)

    lock = threading.Lock()
    active_repositories = set()
    overlap = []
    backend_by_repository = {}
    maximum_active = [0]

    def run_configuration(configuration, backend_container):
        repository = configuration.environment.repository
        with lock:
            if repository in active_repositories:
                overlap.append(repository)
            active_repositories.add(repository)
            backend_by_repository.setdefault(repository, set()).add(backend_container)
            maximum_active[0] = max(maximum_active[0], len(active_repositories))
        time.sleep(0.002)
        with lock:
            active_repositories.remove(repository)
        return "success"

    monkeypatch.setattr(runner, "_run_configuration", run_configuration)

    assert runner.run() == 0
    assert overlap == []
    assert maximum_active[0] == 2
    assert all(len(backends) == 1 for backends in backend_by_repository.values())
    assert set().union(*backend_by_repository.values()) == {
        "backend-worker-1",
        "backend-worker-2",
    }
