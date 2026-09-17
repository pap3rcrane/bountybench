import json
import threading
import time
from argparse import Namespace
from pathlib import Path

from scripts.run_teacher_prompt_matrix import (
    COMPARATIVE_RANKING_STUDENT_PROMPT,
    DEFAULT_PHASE_ITERATIONS,
    Environment,
    MATRIX_BACKEND_CONTAINERS,
    MatrixRunner,
    TEACHER_MAX_INPUT_TOKENS,
    TEACHER_MAX_OUTPUT_TOKENS,
    build_configurations,
    create_parser,
    load_excluded_configuration_keys,
    load_successful_configuration_keys,
    parse_runner_event,
    parse_runner_events,
)

PROMPT_FILE = "prompts/system_prompts/optimizer.txt"


def _args(*, mode):
    return Namespace(
        matrix_name=f"test_{mode}",
        teacher_mode=mode,
        prompt_placement="both",
        teacher_max_input_tokens=TEACHER_MAX_INPUT_TOKENS,
        teacher_max_output_tokens=TEACHER_MAX_OUTPUT_TOKENS,
        phase_iterations=DEFAULT_PHASE_ITERATIONS,
        skip_configurations_from=[],
        exclude_configurations_from=[],
        prompt_file=[PROMPT_FILE],
        environment=[Environment("lunary", "0", "detect_workflow")],
        launcher="run_teacher_matrix.sh",
        launcher_argument=["--dry-run"],
        dry_run=False,
        prune_dind_between_repositories=False,
        verbose=False,
        no_progress=True,
        jobs=1,
        setup_jobs=1,
    )


def test_teacher_token_limits_default_to_gemini_maxima_and_reach_workflow(
    tmp_path, monkeypatch
):
    parser = create_parser()
    parsed = parser.parse_args(
        [
            "--matrix-name",
            "limits",
            "--teacher-mode",
            "observe",
            "--prompt-file",
            PROMPT_FILE,
            "--environment",
            "lunary|0|detect_workflow",
            "--launcher",
            "run_teacher_matrix.sh",
        ]
    )
    assert parsed.teacher_max_input_tokens == 1048576
    assert parsed.teacher_max_output_tokens == 65536
    assert parsed.phase_iterations == 300

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    runner = MatrixRunner(parsed)
    try:
        command = runner._command(runner.configurations[0], "backend-service")
    finally:
        runner.progress.close()
        runner.recorder.close()

    input_index = command.index("--teacher_max_input_tokens")
    output_index = command.index("--teacher_max_output_tokens")
    assert command[input_index + 1] == "1048576"
    assert command[output_index + 1] == "65536"
    phase_index = command.index("--phase_iterations")
    assert command[phase_index + 1] == "300"


def test_custom_phase_iterations_reach_every_student_workflow(tmp_path, monkeypatch):
    parser = create_parser()
    parsed = parser.parse_args(
        [
            "--matrix-name",
            "iterations",
            "--teacher-mode",
            "observe",
            "--phase-iterations",
            "17",
            "--prompt-file",
            PROMPT_FILE,
            "--environment",
            "lunary|0|detect_workflow",
            "--launcher",
            "run_teacher_matrix.sh",
        ]
    )
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    runner = MatrixRunner(parsed)
    try:
        command = runner._command(runner.configurations[0], "backend-service")
    finally:
        runner.progress.close()
        runner.recorder.close()

    phase_index = command.index("--phase_iterations")
    assert command[phase_index + 1] == "17"
    assert runner.progress.phase_iterations == 17


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
    assert len(MATRIX_BACKEND_CONTAINERS) == 30
    assert len(environments) == 9

    system_only = build_configurations(
        environments=[environments[0]],
        prompt_files=prompts,
        repetitions=5,
        placements=("system",),
    )
    assert len(system_only) == 30


def test_user_prompt_placement_runs_only_prepend_and_baseline():
    configurations = build_configurations(
        environments=[Environment("lunary", "0", "detect_workflow")],
        prompt_files=[PROMPT_FILE],
        repetitions=5,
        placements=("prepend",),
    )

    assert len(configurations) == 10
    assert {item.placement for item in configurations} == {"prepend", "none"}


def test_filtered_prompt_matrix_can_omit_baseline():
    configurations = build_configurations(
        environments=[Environment("lunary", "0", "detect_workflow")],
        prompt_files=["prompts/system_prompts/comparative_ranking.txt"],
        repetitions=5,
        placements=("system",),
        include_baseline=False,
    )

    assert len(configurations) == 5
    assert {item.system_prompt_name for item in configurations} == {
        "comparative_ranking"
    }
    assert {item.placement for item in configurations} == {"system"}


def test_comparative_ranking_appends_candidate_requirement_to_student_command(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    args = _args(mode="steer")
    args.prompt_placement = "system"
    args.prompt_file = ["prompts/system_prompts/comparative_ranking.txt"]
    args.exclude_baseline = True
    runner = MatrixRunner(args)
    try:
        command = runner._command(runner.configurations[0], "backend-service")
    finally:
        runner.progress.close()
        runner.recorder.close()

    append_index = command.index("--student_prompt_append_file")
    assert command[append_index + 1] == COMPARATIVE_RANKING_STUDENT_PROMPT


def test_non_comparative_prompt_does_not_change_student_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    runner = MatrixRunner(_args(mode="observe"))
    try:
        command = runner._command(runner.configurations[0], "backend-service")
    finally:
        runner.progress.close()
        runner.recorder.close()

    assert "--student_prompt_append_file" not in command


def test_runner_event_parser_ignores_normal_output():
    assert parse_runner_event("Finished iteration 1") is None
    assert parse_runner_event(_event("student_run_started", run_role="normal")) == {
        "event": "student_run_started",
        "timestamp": "2026-09-14T12:00:00-04:00",
        "run_role": "normal",
    }


def test_runner_event_parser_accepts_concatenated_output():
    event = _event("student_run_finished", run_role="normal", status="success")

    assert parse_runner_event(f"shell output without newline{event}") == {
        "event": "student_run_finished",
        "timestamp": "2026-09-14T12:00:00-04:00",
        "run_role": "normal",
        "status": "success",
    }


def test_runner_event_parser_ignores_suffix_and_finds_multiple_events():
    started = _event("student_run_started", run_role="normal")
    finished = _event("student_run_finished", run_role="normal", status="success")

    assert parse_runner_events(f"prefix{started}suffix{finished}trailing output") == [
        {
            "event": "student_run_started",
            "timestamp": "2026-09-14T12:00:00-04:00",
            "run_role": "normal",
        },
        {
            "event": "student_run_finished",
            "timestamp": "2026-09-14T12:00:00-04:00",
            "run_role": "normal",
            "status": "success",
        },
    ]


def test_runner_event_parser_skips_malformed_marker_before_valid_event():
    valid = _event("student_run_started", run_role="normal")

    assert parse_runner_events(f"BOUNTYBENCH_EVENT not-json {valid}") == [
        {
            "event": "student_run_started",
            "timestamp": "2026-09-14T12:00:00-04:00",
            "run_role": "normal",
        }
    ]


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


def test_explicit_exclusion_source_accepts_failed_configurations(tmp_path):
    status_path = tmp_path / "excluded.jsonl"
    record = {
        "record_type": "configuration",
        "configuration_id": "gemini-json-failure",
        "status": "failure",
        "teacher_type": "objective_rewrite",
        "repo_name": "lunary",
        "bounty_number": "0",
        "workflow_type": "detect_workflow",
        "system_prompt_name": "weakness_targeting_task",
        "system_prompt_placement": "system",
        "run_number": 3,
    }
    status_path.write_text(json.dumps(record) + "\n")

    assert load_successful_configuration_keys([str(status_path)]) == set()
    assert load_excluded_configuration_keys([str(status_path)]) == {
        (
            "objective_rewrite",
            "lunary",
            "0",
            "detect_workflow",
            "weakness_targeting_task",
            "system",
            3,
        )
    }


def test_skip_matching_does_not_depend_on_configuration_sequence(tmp_path, monkeypatch):
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


def test_explicit_worker_set_uses_isolated_artifact_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    args = _args(mode="observe")
    args.jobs = 2
    args.backend_container = ["backend-worker-10", "backend-worker-11"]
    args.backend_log_root = [
        ("backend-worker-10", tmp_path / "worker-10"),
        ("backend-worker-11", tmp_path / "worker-11"),
    ]

    runner = MatrixRunner(args)
    try:
        assert runner.backend_containers == [
            "backend-worker-10",
            "backend-worker-11",
        ]
        assert runner._host_log_path("/app/logs/test.json", "backend-worker-10") == str(
            tmp_path / "worker-10/logs/test.json"
        )
        assert runner._host_log_path("full_logs/test.json", "backend-worker-11") == str(
            tmp_path / "worker-11/full_logs/test.json"
        )
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
    batch_start = next(
        record for record in records if record["record_type"] == "batch_start"
    )

    assert len(configurations) == 15
    assert len(students) == 15
    assert batch_start["worker_scope"] == "configuration"
    assert batch_start["scheduling_order"] == "environment_ordered"
    assert batch_start["concurrent_repo_setups"] == 1
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


def test_prunes_dind_once_after_each_environment(tmp_path, monkeypatch):
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

    def run_configuration(configuration, backend_container):
        time.sleep(0.001)
        return "success"

    monkeypatch.setattr(runner, "_run_configuration", run_configuration)
    pruned_environments = []
    monkeypatch.setattr(
        runner,
        "_prune_dind",
        lambda environment_label, backend_container: pruned_environments.append(
            (environment_label, backend_container)
        ),
    )

    assert runner.run() == 0
    assert pruned_environments == [
        ("repo-a bounty 0 · detect_workflow", "backend-service"),
        ("repo-a bounty 1 · patch_workflow", "backend-service"),
        ("repo-b bounty 0 · exploit_workflow", "backend-service"),
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


def test_parallel_workers_consume_environment_ordered_configuration_queue(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    args = _args(mode="observe")
    args.jobs = 3
    args.prompt_placement = "system"
    args.prompt_file = [PROMPT_FILE]
    args.environment = [
        Environment("repo-a", "0", "detect_workflow"),
        Environment("repo-b", "0", "exploit_workflow"),
    ]
    runner = MatrixRunner(args)
    monkeypatch.setattr(runner, "_preflight", lambda: None)

    lock = threading.Lock()
    active_environments = {}
    start_order = []
    tail_overlap = []
    backend_by_environment = {}
    runs_by_environment = {}
    maximum_active = [0]

    def run_configuration(configuration, backend_container):
        environment = configuration.environment
        with lock:
            start_order.append(environment)
            active_environments[environment] = (
                active_environments.get(environment, 0) + 1
            )
            if (
                environment.repository == "repo-b"
                and active_environments.get(args.environment[0], 0) > 0
            ):
                tail_overlap.append(True)
            backend_by_environment.setdefault(environment, set()).add(backend_container)
            runs_by_environment[environment] = (
                runs_by_environment.get(environment, 0) + 1
            )
            maximum_active[0] = max(
                maximum_active[0], sum(active_environments.values())
            )
        time.sleep(0.002)
        with lock:
            active_environments[environment] -= 1
        return "success"

    monkeypatch.setattr(runner, "_run_configuration", run_configuration)

    assert runner.run() == 0
    assert maximum_active[0] == 3
    assert start_order == [args.environment[0]] * 10 + [args.environment[1]] * 10
    assert tail_overlap
    assert set(runs_by_environment.values()) == {10}
    assert all(len(backends) > 1 for backends in backend_by_environment.values())
    assert set().union(*backend_by_environment.values()) == {
        "backend-worker-1",
        "backend-worker-2",
        "backend-worker-3",
    }


def test_parallel_pruning_is_scoped_to_each_worker_transition(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BATCH_LOG_ROOT", str(tmp_path / "batch_logs"))
    args = _args(mode="observe")
    args.jobs = 2
    args.prompt_placement = "system"
    args.environment = [
        Environment("repo-a", "0", "detect_workflow"),
        Environment("repo-b", "0", "exploit_workflow"),
    ]
    args.prune_dind_between_repositories = True
    runner = MatrixRunner(args)
    monkeypatch.setattr(runner, "_preflight", lambda: None)

    def run_configuration(configuration, backend_container):
        time.sleep(0.001)
        return "success"

    monkeypatch.setattr(runner, "_run_configuration", run_configuration)

    lock = threading.Lock()
    pruned_by_backend = {}

    def record_prune(environment_label, backend_container):
        with lock:
            pruned_by_backend.setdefault(backend_container, []).append(
                environment_label
            )

    monkeypatch.setattr(runner, "_prune_dind", record_prune)

    assert runner.run() == 0
    assert pruned_by_backend == {
        "backend-worker-1": [
            "repo-a bounty 0 · detect_workflow",
            "repo-b bounty 0 · exploit_workflow",
        ],
        "backend-worker-2": [
            "repo-a bounty 0 · detect_workflow",
            "repo-b bounty 0 · exploit_workflow",
        ],
    }
