#!/usr/bin/env python3
"""Run a teacher prompt matrix with progress bars and compact status records."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, Optional, TextIO

from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_EVENT_PREFIX = "BOUNTYBENCH_EVENT "
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ITERATION_PATTERN = re.compile(r"Finished iteration (\d+) of .* with (\S+)")
ERROR_PATTERNS = (
    re.compile(r"Initialization error:\s*(.+)"),
    re.compile(r"Error in .* workflow:\s*(.+)"),
    re.compile(r"Error response from daemon:\s*(.+)"),
    re.compile(r"^(?:[A-Za-z_][\w.]*)(?:Error|Exception):\s*(.+)"),
)

STUDENT_MODEL = "openrouter/deepseek/deepseek-chat-v3-0324"
TEACHER_MODEL = "google/gemini-3.6-flash"
PRIMARY_BACKEND_CONTAINER = "backend-service"
MATRIX_BACKEND_CONTAINERS = tuple(f"backend-worker-{number}" for number in range(1, 31))
DEFAULT_CONFIGURATION_WORKERS = 30
DEFAULT_REPO_SETUP_WORKERS = 5
DEFAULT_PHASE_ITERATIONS = 300
REPETITIONS = 5
MAX_INPUT_TOKENS = 1048576
MAX_OUTPUT_TOKENS = 65536
TEACHER_MAX_INPUT_TOKENS = 1048576
TEACHER_MAX_OUTPUT_TOKENS = 65536
COMPARATIVE_RANKING_STUDENT_PROMPT = (
    "prompts/student_prompts/comparative_ranking_candidates.txt"
)
PLACEMENTS = ("prepend", "system")
PROMPT_PLACEMENT_MAP = {
    "user": ("prepend",),
    "system": ("system",),
    "both": PLACEMENTS,
}
OBJECTIVE_RUN_ROLES = (
    "source_1",
    "source_2",
    "source_3",
    "rewritten_objective",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def compact_error(error: Optional[str]) -> Optional[str]:
    if not error:
        return None
    return " ".join(error.split())[:1000]


@dataclass(frozen=True)
class Environment:
    repository: str
    bounty_number: str
    workflow_type: str

    @property
    def label(self) -> str:
        return f"{self.repository} bounty {self.bounty_number} · {self.workflow_type}"


@dataclass(frozen=True)
class Configuration:
    sequence: int
    configuration_id: str
    environment: Environment
    prompt_file: Optional[str]
    system_prompt_name: str
    placement: str
    repetition: int


@dataclass
class StudentRunState:
    role: str
    backend_container: Optional[str] = None
    status: str = "planned"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    started_monotonic: Optional[float] = None
    duration_seconds: Optional[float] = None
    workflow_log_path: Optional[str] = None
    error: Optional[str] = None
    recorded: bool = False


class JsonlRecorder:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("w", buffering=1)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock:
            self._file.write(json.dumps(record, default=str, sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


class ProgressDisplay:
    def __init__(
        self,
        *,
        matrix_name: str,
        total: int,
        enabled: bool,
        workers: list[str],
        phase_iterations: int,
    ):
        self.enabled = enabled
        self._lock = threading.RLock()
        self.active = {}
        self.worker_positions = {
            worker: position for position, worker in enumerate(workers, start=1)
        }
        self.worker_labels = {
            worker: f"Job {position}"
            for position, worker in enumerate(workers, start=1)
        }
        self.succeeded = 0
        self.failed = 0
        self.skipped = 0
        self.phase_iterations = phase_iterations
        self.overall = tqdm(
            total=total,
            desc=f"Overall {matrix_name}",
            unit="config",
            dynamic_ncols=True,
            smoothing=0.2,
            disable=not enabled,
            position=0,
        )

    def write(self, message: str) -> None:
        with self._lock:
            if self.enabled:
                tqdm.write(message)
            else:
                print(message, flush=True)

    def _replace_active(
        self, worker: str, *, description: str, total: Optional[int]
    ) -> None:
        self._close_active(worker)
        if not self.enabled:
            return
        label = self.worker_labels[worker]
        if total is None:
            self.active[worker] = tqdm(
                desc=f"{label} · {description}",
                unit="",
                bar_format="{desc} [{elapsed}]",
                dynamic_ncols=True,
                leave=False,
                position=self.worker_positions[worker],
            )
        else:
            self.active[worker] = tqdm(
                total=total,
                desc=f"{label} · {description}",
                unit="step",
                dynamic_ncols=True,
                smoothing=0.2,
                leave=False,
                position=self.worker_positions[worker],
            )

    def start_configuration(
        self, configuration: Configuration, total: int, worker: str
    ) -> None:
        environment = configuration.environment
        label = (
            f"{environment.repository} bounty {environment.bounty_number} · "
            f"{environment.workflow_type} · {configuration.system_prompt_name} · "
            f"{configuration.placement} · repeat {configuration.repetition}/{REPETITIONS}"
        )
        with self._lock:
            if self.enabled:
                self._replace_active(worker, description=label, total=None)
            else:
                self.write(
                    f"[{configuration.sequence}/{total}] "
                    f"{self.worker_labels[worker]} · {label}"
                )

    def start_student(self, role: str, worker: str) -> None:
        with self._lock:
            self._replace_active(
                worker, description=f"Student {role}", total=self.phase_iterations
            )

    def finish_student(self, worker: str) -> None:
        with self._lock:
            self._close_active(worker)

    def start_teacher_rewrite(self, worker: str) -> None:
        with self._lock:
            self._replace_active(
                worker, description="Teacher objective rewrite", total=None
            )

    def finish_teacher_rewrite(self, worker: str) -> None:
        with self._lock:
            self._close_active(worker)

    def update_iteration(self, iteration: int, agent_id: str, worker: str) -> None:
        with self._lock:
            active = self.active.get(worker)
            if not self.enabled or active is None or active.total is None:
                return
            completed = min(iteration + 1, active.total)
            if completed > active.n:
                active.update(completed - active.n)
            active.set_postfix_str(f"agent={agent_id}", refresh=True)

    def heartbeat(self, worker: str) -> None:
        with self._lock:
            active = self.active.get(worker)
            if self.enabled and active is not None:
                active.refresh()
            if self.enabled:
                self.overall.refresh()

    def finish_configuration(self, status: str, worker: str) -> None:
        with self._lock:
            self._close_active(worker)
            if status == "success":
                self.succeeded += 1
            elif status == "failure":
                self.failed += 1
            else:
                self.skipped += 1
            if self.enabled:
                self.overall.update(1)
                self.overall.set_postfix_str(
                    f"success={self.succeeded} failure={self.failed} "
                    f"skipped={self.skipped}",
                    refresh=True,
                )

    def _close_active(self, worker: str) -> None:
        active = self.active.pop(worker, None)
        if active is not None:
            active.close()

    def close(self) -> None:
        with self._lock:
            for worker in list(self.active):
                self._close_active(worker)
            self.overall.close()


def parse_environment(value: str) -> Environment:
    parts = value.split("|")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Environment must have the form REPOSITORY|BOUNTY|WORKFLOW"
        )
    repository, bounty_number, workflow_type = parts
    if workflow_type not in {
        "detect_workflow",
        "exploit_workflow",
        "patch_workflow",
    }:
        raise argparse.ArgumentTypeError(f"Unsupported workflow: {workflow_type}")
    return Environment(repository, bounty_number, workflow_type)


def parse_backend_log_root(value: str) -> tuple[str, Path]:
    container, separator, path = value.partition("=")
    if not separator or not container or not path:
        raise argparse.ArgumentTypeError(
            "Backend log root must have the form CONTAINER=PATH"
        )
    return container, Path(path).expanduser().resolve()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_configurations(
    *,
    environments: Iterable[Environment],
    prompt_files: Iterable[str],
    repetitions: int,
    placements: Iterable[str] = PLACEMENTS,
    include_baseline: bool = True,
) -> list[Configuration]:
    configurations = []
    sequence = 0
    for environment in environments:
        for prompt_file in prompt_files:
            prompt_name = Path(prompt_file).stem
            for placement in placements:
                for repetition in range(1, repetitions + 1):
                    sequence += 1
                    configuration_id = (
                        f"{sequence:04d}_{environment.repository}_bounty_"
                        f"{environment.bounty_number}_{environment.workflow_type}_"
                        f"{prompt_name}_{placement}_repeat_{repetition}"
                    )
                    configurations.append(
                        Configuration(
                            sequence=sequence,
                            configuration_id=re.sub(
                                r"[^a-zA-Z0-9_.-]", "_", configuration_id
                            ),
                            environment=environment,
                            prompt_file=prompt_file,
                            system_prompt_name=prompt_name,
                            placement=placement,
                            repetition=repetition,
                        )
                    )
        if include_baseline:
            for repetition in range(1, repetitions + 1):
                sequence += 1
                configuration_id = (
                    f"{sequence:04d}_{environment.repository}_bounty_"
                    f"{environment.bounty_number}_{environment.workflow_type}_"
                    f"none_none_repeat_{repetition}"
                )
                configurations.append(
                    Configuration(
                        sequence=sequence,
                        configuration_id=re.sub(
                            r"[^a-zA-Z0-9_.-]", "_", configuration_id
                        ),
                        environment=environment,
                        prompt_file=None,
                        system_prompt_name="none",
                        placement="none",
                        repetition=repetition,
                    )
                )
    return configurations


def parse_runner_events(line: str) -> list[dict]:
    clean_line = ANSI_ESCAPE.sub("", line).strip()
    events = []
    search_from = 0
    decoder = json.JSONDecoder()
    while True:
        marker_index = clean_line.find(RUNNER_EVENT_PREFIX, search_from)
        if marker_index < 0:
            break
        payload_start = marker_index + len(RUNNER_EVENT_PREFIX)
        payload = clean_line[payload_start:].lstrip()
        try:
            event, payload_length = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            search_from = payload_start
            continue
        if isinstance(event, dict):
            events.append(event)
        search_from = payload_start + (len(clean_line[payload_start:]) - len(payload)) + payload_length
    return events


def parse_runner_event(line: str) -> Optional[dict]:
    events = parse_runner_events(line)
    return events[0] if events else None


def extract_error(line: str) -> Optional[str]:
    clean_line = ANSI_ESCAPE.sub("", line).strip()
    for pattern in ERROR_PATTERNS:
        match = pattern.search(clean_line)
        if match:
            return compact_error(match.group(1))
    return None


def load_successful_configuration_keys(status_paths: Iterable[str]) -> set[tuple]:
    """Load only finalized, successful top-level configurations."""
    configuration_keys = set()
    for supplied_path in status_paths:
        path = Path(supplied_path).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if not path.is_file():
            raise ValueError(f"Skip status file does not exist: {path}")

        with path.open() as status_file:
            for line_number, line in enumerate(status_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in skip status file {path}:{line_number}"
                    ) from error
                if (
                    record.get("record_type") == "configuration"
                    and record.get("status") == "success"
                ):
                    required_fields = (
                        "teacher_type",
                        "repo_name",
                        "bounty_number",
                        "workflow_type",
                        "system_prompt_name",
                        "system_prompt_placement",
                        "run_number",
                    )
                    if not all(field in record for field in required_fields):
                        raise ValueError(
                            f"Successful configuration in {path}:{line_number} "
                            "is missing fields required for safe matching"
                        )
                    configuration_keys.add(
                        tuple(record[field] for field in required_fields)
                    )
    return configuration_keys


def configuration_key(configuration: Configuration, teacher_type: str) -> tuple:
    environment = configuration.environment
    return (
        teacher_type,
        environment.repository,
        environment.bounty_number,
        environment.workflow_type,
        configuration.system_prompt_name,
        configuration.placement,
        configuration.repetition,
    )


class MatrixRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        supplied_backends = list(getattr(args, "backend_container", []))
        if supplied_backends:
            if len(supplied_backends) != args.jobs:
                raise ValueError(
                    "--jobs must equal the number of --backend-container values"
                )
            if len(set(supplied_backends)) != len(supplied_backends):
                raise ValueError("Backend container names must be unique")
            self.backend_containers = supplied_backends
        else:
            self.backend_containers = (
                [PRIMARY_BACKEND_CONTAINER]
                if args.jobs == 1
                else list(MATRIX_BACKEND_CONTAINERS[: args.jobs])
            )
        supplied_log_roots = dict(getattr(args, "backend_log_root", []))
        unknown_log_roots = set(supplied_log_roots) - set(self.backend_containers)
        if unknown_log_roots:
            raise ValueError(
                "Log roots supplied for unknown backend containers: "
                + ", ".join(sorted(unknown_log_roots))
            )
        self.backend_log_roots = supplied_log_roots
        self.environments = args.environment
        self.prompt_files = args.prompt_file
        self.prompt_placement = getattr(args, "prompt_placement", "both")
        self.placements = PROMPT_PLACEMENT_MAP[self.prompt_placement]
        self.skip_configuration_sources = list(
            getattr(args, "skip_configurations_from", [])
        )
        requested_skip_keys = load_successful_configuration_keys(
            self.skip_configuration_sources
        )
        self.configurations = build_configurations(
            environments=self.environments,
            prompt_files=self.prompt_files,
            repetitions=REPETITIONS,
            placements=self.placements,
            include_baseline=not getattr(args, "exclude_baseline", False),
        )
        self.skip_configuration_ids = {
            configuration.configuration_id
            for configuration in self.configurations
            if configuration_key(configuration, args.teacher_mode)
            in requested_skip_keys
        }
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f%z")
        self.batch_id = f"{args.matrix_name}_{timestamp}"
        batch_log_root = Path(
            os.environ.get("BATCH_LOG_ROOT", REPOSITORY_ROOT / "batch_logs")
        )
        runs_root = Path(os.environ.get("RUNS_ROOT", REPOSITORY_ROOT / "runs"))
        self.batch_log_dir = batch_log_root / args.matrix_name / timestamp
        self.run_dir = runs_root / args.matrix_name / timestamp
        self.status_path = self.run_dir / "run_status.jsonl"
        self.batch_log_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = JsonlRecorder(self.status_path)
        self.progress = ProgressDisplay(
            matrix_name=args.matrix_name,
            total=len(self.configurations),
            enabled=not args.no_progress,
            workers=self.backend_containers,
            phase_iterations=args.phase_iterations,
        )
        self._state_lock = threading.Lock()
        self._active_processes = {}
        self._stop_event = threading.Event()
        self.failed_configurations = 0
        self.completed_configurations = 0

    @property
    def planned_roles(self) -> tuple[str, ...]:
        if self.args.teacher_mode == "objective_rewrite":
            return OBJECTIVE_RUN_ROLES
        return ("normal",)

    def _base_record(self, configuration: Configuration) -> dict:
        environment = configuration.environment
        return {
            "batch_id": self.batch_id,
            "configuration_id": configuration.configuration_id,
            "sequence": configuration.sequence,
            "repo_name": environment.repository,
            "bounty_number": environment.bounty_number,
            "workflow_type": environment.workflow_type,
            "system_prompt_name": configuration.system_prompt_name,
            "system_prompt_file": configuration.prompt_file,
            "system_prompt_placement": configuration.placement,
            "run_number": configuration.repetition,
            "teacher_type": self.args.teacher_mode,
            "teacher_model": TEACHER_MODEL,
            "student_model": STUDENT_MODEL,
        }

    def _batch_record(self, record_type: str, **fields) -> dict:
        expected_student_runs = len(self.configurations) * len(self.planned_roles)
        return {
            "record_type": record_type,
            "batch_id": self.batch_id,
            "matrix_name": self.args.matrix_name,
            "teacher_type": self.args.teacher_mode,
            "launcher": self.args.launcher,
            "launcher_arguments": self.args.launcher_argument,
            "invocation": shlex.join(sys.argv),
            "student_model": STUDENT_MODEL,
            "teacher_model": TEACHER_MODEL,
            "teacher_max_input_tokens": self.args.teacher_max_input_tokens,
            "teacher_max_output_tokens": self.args.teacher_max_output_tokens,
            "phase_iterations": self.args.phase_iterations,
            "repetitions": REPETITIONS,
            "concurrent_jobs": self.args.jobs,
            "concurrent_repo_setups": self.args.setup_jobs,
            "worker_scope": "configuration",
            "scheduling_order": "environment_ordered",
            "backend_containers": self.backend_containers,
            "backend_log_roots": {
                backend: str(path) for backend, path in self.backend_log_roots.items()
            },
            "prompt_placement_selection": self.prompt_placement,
            "placements": list(self.placements) + ["none"],
            "skip_configuration_sources": self.skip_configuration_sources,
            "skip_configuration_count": len(self.skip_configuration_ids),
            "prompt_files": self.prompt_files,
            "environments": [
                {
                    "repo_name": environment.repository,
                    "bounty_number": environment.bounty_number,
                    "workflow_type": environment.workflow_type,
                }
                for environment in self.environments
            ],
            "configuration_count": len(self.configurations),
            "expected_student_run_count": expected_student_runs,
            "batch_log_directory": str(self.batch_log_dir),
            "status_file": str(self.status_path),
            "dry_run": self.args.dry_run,
            "prune_dind_between_repositories": (
                self.args.prune_dind_between_repositories
            ),
            **fields,
        }

    def _student_record(
        self, configuration: Configuration, state: StudentRunState
    ) -> dict:
        return {
            "record_type": "student_run",
            **self._base_record(configuration),
            "parent_configuration_id": configuration.configuration_id,
            "run_role": state.role,
            "status": state.status,
            "exit_code": (
                0
                if state.status == "success"
                else 1 if state.status == "failure" else None
            ),
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "duration_seconds": state.duration_seconds,
            "workflow_log_path": state.workflow_log_path,
            "configuration_log_path": str(
                self.batch_log_dir / f"{configuration.configuration_id}.log"
            ),
            "backend_container": state.backend_container,
            "error": state.error,
        }

    def _record_student(
        self, configuration: Configuration, state: StudentRunState
    ) -> None:
        if state.recorded:
            return
        self.recorder.write(self._student_record(configuration, state))
        state.recorded = True

    def _configuration_record(
        self,
        configuration: Configuration,
        *,
        status: str,
        exit_code: Optional[int],
        started_at: str,
        finished_at: str,
        duration_seconds: float,
        error: Optional[str],
        backend_container: str,
    ) -> dict:
        return {
            "record_type": "configuration",
            **self._base_record(configuration),
            "parent_configuration_id": None,
            "run_role": "configuration",
            "status": status,
            "exit_code": exit_code,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "workflow_log_path": None,
            "configuration_log_path": str(
                self.batch_log_dir / f"{configuration.configuration_id}.log"
            ),
            "backend_container": backend_container,
            "error": error,
        }

    def _host_log_path(
        self, path: Optional[str], backend_container: str
    ) -> Optional[str]:
        if not path:
            return None
        candidate = Path(path)
        worker_root = self.backend_log_roots.get(backend_container)
        if worker_root is not None:
            if candidate.is_absolute() and str(candidate).startswith("/app/logs/"):
                return str(worker_root / "logs" / candidate.relative_to("/app/logs"))
            if candidate.is_absolute() and str(candidate).startswith("/app/full_logs/"):
                return str(
                    worker_root / "full_logs" / candidate.relative_to("/app/full_logs")
                )
            if not candidate.is_absolute() and candidate.parts:
                if candidate.parts[0] in {"logs", "full_logs"}:
                    return str(worker_root / candidate)
        if candidate.is_absolute() and str(candidate).startswith("/app/"):
            candidate = REPOSITORY_ROOT / candidate.relative_to("/app")
        elif not candidate.is_absolute():
            candidate = REPOSITORY_ROOT / candidate
        return str(candidate)

    def _command(
        self, configuration: Configuration, backend_container: str
    ) -> list[str]:
        environment = configuration.environment
        command = [
            "docker",
            "exec",
            "-i",
            backend_container,
            "python",
            "-m",
            "workflows.runner",
            "--workflow-type",
            environment.workflow_type,
            "--task_dir",
            f"bountytasks/{environment.repository}",
            "--bounty_number",
            environment.bounty_number,
            "--model",
            STUDENT_MODEL,
            "--teacher_model",
            TEACHER_MODEL,
            "--teacher_mode",
            self.args.teacher_mode,
            "--teacher_system_prompt_placement",
            configuration.placement,
            "--teacher_max_input_tokens",
            str(self.args.teacher_max_input_tokens),
            "--teacher_max_output_tokens",
            str(self.args.teacher_max_output_tokens),
            "--phase_iterations",
            str(self.args.phase_iterations),
            "--max_input_tokens",
            str(MAX_INPUT_TOKENS),
            "--max_output_tokens",
            str(MAX_OUTPUT_TOKENS),
            "--logging_level",
            "DEBUG",
        ]
        if configuration.prompt_file:
            command.extend(["--teacher_system_prompt_file", configuration.prompt_file])
        if configuration.system_prompt_name == "comparative_ranking":
            command.extend(
                [
                    "--student_prompt_append_file",
                    COMPARATIVE_RANKING_STUDENT_PROMPT,
                ]
            )
        if self.args.teacher_mode == "objective_rewrite":
            command.append("--generate_source_runs")
        return command

    def _wait_for_dind(self, backend_container: str, timeout: int = 180) -> None:
        self.progress.write(f"Waiting for Docker-in-Docker in {backend_container}...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "exec", backend_container, "docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return
            time.sleep(2)
        raise RuntimeError(
            f"Docker-in-Docker did not become ready in '{backend_container}' "
            f"within {timeout} seconds."
        )

    def _preflight(self) -> None:
        for prompt_file in self.prompt_files:
            path = REPOSITORY_ROOT / prompt_file
            if not path.is_file():
                raise RuntimeError(f"Missing system prompt: {path}")
        if any(Path(path).stem == "comparative_ranking" for path in self.prompt_files):
            student_prompt_path = REPOSITORY_ROOT / COMPARATIVE_RANKING_STUDENT_PROMPT
            if not student_prompt_path.is_file():
                raise RuntimeError(f"Missing student prompt: {student_prompt_path}")

        for backend_container in self.backend_containers:
            inspect = subprocess.run(
                ["docker", "inspect", backend_container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if inspect.returncode != 0:
                raise RuntimeError(
                    f"Container '{backend_container}' was not found. Start the "
                    "matrix through ./run_teacher_matrix.sh."
                )

            running = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", backend_container],
                capture_output=True,
                text=True,
                check=False,
            )
            if running.returncode != 0 or running.stdout.strip() != "true":
                raise RuntimeError(f"Container '{backend_container}' is not running.")

            self._wait_for_dind(backend_container)

            feature_check = subprocess.run(
                [
                    "docker",
                    "exec",
                    backend_container,
                    "python",
                    "-c",
                    (
                        "from agents.teacher_agent import "
                        "TeacherSystemPromptPlacement; "
                        "from resources.model_resource.model_mapping import "
                        "get_model_info; "
                        "from resources.model_resource.model_resource import "
                        "ModelResourceConfig; "
                        "from workflows.runner import RUNNER_EVENT_PREFIX; "
                        "assert TeacherSystemPromptPlacement.NONE.value == 'none'; "
                        "assert RUNNER_EVENT_PREFIX == 'BOUNTYBENCH_EVENT '; "
                        "assert ModelResourceConfig(model='mock', "
                        "use_mock_model=True, preserve_oldest_input=True)"
                        ".preserve_oldest_input; "
                        "get_model_info('google/gemini-3.6-flash', helm=False)"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if feature_check.returncode != 0:
                raise RuntimeError(
                    f"'{backend_container}' does not contain the current teacher "
                    "code. Rebuild with: docker compose up -d --build "
                    "--force-recreate backend"
                )

            for prompt_file in self.prompt_files:
                mounted_prompt = f"/app/{prompt_file}"
                readable = subprocess.run(
                    ["docker", "exec", backend_container, "test", "-r", mounted_prompt],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if readable.returncode != 0:
                    raise RuntimeError(
                        f"Prompt is not mounted in '{backend_container}': "
                        f"{mounted_prompt}. Rebuild and rerun the launcher."
                    )

    def _prune_dind(self, environment_label: str, backend_container: str) -> None:
        """Remove unused DinD resources without deleting tagged images."""
        self.progress.write(
            f"Pruning unused DinD resources after {environment_label} "
            f"in {backend_container}..."
        )
        result = subprocess.run(
            [
                "docker",
                "exec",
                backend_container,
                "docker",
                "system",
                "prune",
                "-f",
                "--volumes",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error = compact_error(result.stderr or result.stdout) or "Unknown error."
            self.progress.write(
                f"WARNING: DinD cleanup after {environment_label} in "
                f"{backend_container} failed: {error}"
            )
            return

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        summary = lines[-1] if lines else "cleanup completed"
        self.progress.write(
            f"DinD cleanup after {environment_label} in "
            f"{backend_container}: {summary}"
        )

    def _handle_event(
        self,
        *,
        event: dict,
        configuration: Configuration,
        states: dict[str, StudentRunState],
        backend_container: str,
    ) -> Optional[str]:
        event_name = event.get("event")
        role = event.get("run_role")
        event_error = compact_error(event.get("error"))

        if event_name == "student_run_started" and role in states:
            state = states[role]
            if state.status == "planned":
                state.status = "running"
                state.started_at = event.get("timestamp") or now_iso()
                state.started_monotonic = time.monotonic()
            self.progress.start_student(role, backend_container)
        elif event_name == "student_run_finished" and role in states:
            state = states[role]
            if state.started_at is None:
                state.started_at = event.get("timestamp") or now_iso()
                state.started_monotonic = time.monotonic()
            state.finished_at = event.get("timestamp") or now_iso()
            state.status = "success" if event.get("status") == "success" else "failure"
            if state.started_monotonic is not None:
                state.duration_seconds = round(
                    time.monotonic() - state.started_monotonic, 3
                )
            state.workflow_log_path = self._host_log_path(
                event.get("workflow_log_path"), backend_container
            )
            state.error = event_error
            self._record_student(configuration, state)
            self.progress.finish_student(backend_container)
        elif event_name == "teacher_rewrite_started":
            self.progress.start_teacher_rewrite(backend_container)
        elif event_name == "teacher_rewrite_finished":
            self.progress.finish_teacher_rewrite(backend_container)

        return event_error

    def _finalize_unrecorded_students(
        self,
        *,
        configuration: Configuration,
        states: dict[str, StudentRunState],
        process_exit_code: Optional[int],
        configuration_error: Optional[str],
        dry_run: bool = False,
    ) -> None:
        finished_at = now_iso()
        for state in states.values():
            if state.recorded:
                continue
            if dry_run:
                state.status = "skipped"
                state.error = "Dry run; student workflow was not started."
            elif state.status == "running":
                state.status = "failure"
                state.error = configuration_error or (
                    f"Student workflow exited before a completion event "
                    f"(process exit code {process_exit_code})."
                )
            else:
                state.status = "skipped"
                state.error = configuration_error or (
                    "A prerequisite stage failed before this student workflow started."
                )
            state.finished_at = finished_at
            if state.started_monotonic is not None:
                state.duration_seconds = round(
                    time.monotonic() - state.started_monotonic, 3
                )
            else:
                state.duration_seconds = 0.0
            self._record_student(configuration, state)

    def _run_configuration(
        self, configuration: Configuration, backend_container: str
    ) -> str:
        self.progress.start_configuration(
            configuration, len(self.configurations), backend_container
        )
        command = self._command(configuration, backend_container)
        configuration_started_at = now_iso()
        configuration_started_monotonic = time.monotonic()
        states = {
            role: StudentRunState(role=role, backend_container=backend_container)
            for role in self.planned_roles
        }
        configuration_log_path = (
            self.batch_log_dir / f"{configuration.configuration_id}.log"
        )

        if self.args.dry_run:
            self.progress.write("DRY RUN: " + shlex.join(command))
            self._finalize_unrecorded_students(
                configuration=configuration,
                states=states,
                process_exit_code=None,
                configuration_error="Dry run; command was not executed.",
                dry_run=True,
            )
            finished_at = now_iso()
            self.recorder.write(
                self._configuration_record(
                    configuration,
                    status="skipped",
                    exit_code=None,
                    started_at=configuration_started_at,
                    finished_at=finished_at,
                    duration_seconds=round(
                        time.monotonic() - configuration_started_monotonic, 3
                    ),
                    error="Dry run; command was not executed.",
                    backend_container=backend_container,
                )
            )
            self.progress.finish_configuration("skipped", backend_container)
            return "skipped"

        detected_error = None
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._state_lock:
                self._active_processes[configuration.configuration_id] = process
            assert process.stdout is not None
            with configuration_log_path.open("w", buffering=1) as output_log:
                for line in process.stdout:
                    output_log.write(line)
                    if self.args.verbose:
                        self.progress.write(f"[{backend_container}] {line.rstrip()}")
                    for event in parse_runner_events(line):
                        event_error = self._handle_event(
                            event=event,
                            configuration=configuration,
                            states=states,
                            backend_container=backend_container,
                        )
                        detected_error = event_error or detected_error
                    line_error = extract_error(line)
                    detected_error = line_error or detected_error

                    iteration_match = ITERATION_PATTERN.search(
                        ANSI_ESCAPE.sub("", line)
                    )
                    if iteration_match:
                        self.progress.update_iteration(
                            int(iteration_match.group(1)),
                            iteration_match.group(2),
                            backend_container,
                        )
                    else:
                        self.progress.heartbeat(backend_container)
            exit_code = process.wait()
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            detected_error = "Interrupted by user."
            exit_code = 130
        except OSError as error:
            detected_error = compact_error(f"{error.__class__.__name__}: {error}")
            exit_code = 1
        finally:
            with self._state_lock:
                self._active_processes.pop(configuration.configuration_id, None)

        self._finalize_unrecorded_students(
            configuration=configuration,
            states=states,
            process_exit_code=exit_code,
            configuration_error=detected_error,
        )
        all_students_succeeded = all(
            state.status == "success" for state in states.values()
        )
        status = "success" if exit_code == 0 and all_students_succeeded else "failure"
        if status == "failure" and not detected_error:
            detected_error = (
                f"Configuration exited with code {exit_code}; see "
                f"{configuration_log_path}."
            )
        finished_at = now_iso()
        self.recorder.write(
            self._configuration_record(
                configuration,
                status=status,
                exit_code=exit_code,
                started_at=configuration_started_at,
                finished_at=finished_at,
                duration_seconds=round(
                    time.monotonic() - configuration_started_monotonic, 3
                ),
                error=compact_error(detected_error),
                backend_container=backend_container,
            )
        )
        self.progress.finish_configuration(status, backend_container)
        if status == "failure":
            self.progress.write(
                f"FAILED: {configuration.configuration_id}: {detected_error}"
            )
        elif not self.progress.enabled:
            self.progress.write(f"SUCCESS: {configuration.configuration_id}")

        if exit_code == 130:
            raise KeyboardInterrupt
        return status

    def _skip_configuration(
        self, configuration: Configuration, backend_container: str
    ) -> str:
        """Record a prior successful configuration without executing it again."""
        self.progress.start_configuration(
            configuration, len(self.configurations), backend_container
        )
        reason = (
            "Skipped because a supplied status index records this finalized "
            "configuration as successful."
        )
        finished_at = now_iso()
        for role in self.planned_roles:
            state = StudentRunState(
                role=role,
                backend_container=backend_container,
                status="skipped",
                finished_at=finished_at,
                duration_seconds=0.0,
                error=reason,
            )
            self._record_student(configuration, state)
        self.recorder.write(
            self._configuration_record(
                configuration,
                status="skipped",
                exit_code=None,
                started_at=finished_at,
                finished_at=finished_at,
                duration_seconds=0.0,
                error=reason,
                backend_container=backend_container,
            )
        )
        self.progress.finish_configuration("skipped", backend_container)
        return "skipped"

    def _run_configuration_worker(
        self, backend_container: str, configurations: Queue
    ) -> list[str]:
        """Consume the ordered configuration queue on one isolated backend."""
        errors = []
        previous_environment = None
        try:
            while not self._stop_event.is_set():
                try:
                    configuration = configurations.get_nowait()
                except Empty:
                    break

                try:
                    environment = configuration.environment
                    if (
                        previous_environment is not None
                        and environment != previous_environment
                        and self.args.prune_dind_between_repositories
                        and not self.args.dry_run
                    ):
                        self._prune_dind(
                            previous_environment.label, backend_container
                        )
                    previous_environment = environment

                    if configuration.configuration_id in self.skip_configuration_ids:
                        status = self._skip_configuration(
                            configuration, backend_container
                        )
                    else:
                        status = self._run_configuration(
                            configuration, backend_container
                        )
                    with self._state_lock:
                        self.completed_configurations += 1
                        if status == "failure":
                            self.failed_configurations += 1
                except KeyboardInterrupt:
                    self._stop_event.set()
                    raise
                except Exception as error:
                    errors.append(
                        f"{configuration.configuration_id}: "
                        f"{error.__class__.__name__}: {error}"
                    )
                finally:
                    configurations.task_done()
        finally:
            if (
                previous_environment is not None
                and self.args.prune_dind_between_repositories
                and not self.args.dry_run
                and not self._stop_event.is_set()
            ):
                self._prune_dind(previous_environment.label, backend_container)
        return errors

    def _terminate_active_processes(self) -> None:
        with self._state_lock:
            processes = list(self._active_processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _run_configurations(self) -> list[str]:
        """Run configurations FIFO so environment tails overlap intentionally."""
        configurations = Queue()
        for configuration in self.configurations:
            configurations.put(configuration)

        if self.args.jobs == 1 or self.args.dry_run:
            return self._run_configuration_worker(
                self.backend_containers[0], configurations
            )

        executor = ThreadPoolExecutor(
            max_workers=self.args.jobs,
            thread_name_prefix="teacher-matrix",
        )
        futures = {
            executor.submit(
                self._run_configuration_worker,
                backend_container,
                configurations,
            ): backend_container
            for backend_container in self.backend_containers
        }
        errors = []
        try:
            for future in as_completed(futures):
                try:
                    errors.extend(future.result())
                except Exception as error:
                    backend_container = futures[future]
                    errors.append(
                        f"{backend_container}: {error.__class__.__name__}: {error}"
                    )
        except KeyboardInterrupt:
            self._stop_event.set()
            self._terminate_active_processes()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return errors

    def run(self) -> int:
        batch_started_at = now_iso()
        batch_started_monotonic = time.monotonic()
        self.recorder.write(
            self._batch_record(
                "batch_start",
                status="running",
                started_at=batch_started_at,
                finished_at=None,
                duration_seconds=None,
                error=None,
            )
        )
        self.progress.write(f"Status index: {self.status_path}")
        self.progress.write(f"Detailed command output: {self.batch_log_dir}")
        if self.skip_configuration_ids:
            self.progress.write(
                f"Skipping {len(self.skip_configuration_ids)} configurations "
                "previously recorded as successful."
            )

        batch_error = None
        interrupted = False
        try:
            if not self.args.dry_run:
                self._preflight()
            configuration_errors = self._run_configurations()
            if configuration_errors:
                batch_error = compact_error("; ".join(configuration_errors))
        except KeyboardInterrupt:
            interrupted = True
            self._stop_event.set()
            self._terminate_active_processes()
            batch_error = "Interrupted by user."
            self.progress.write("Batch interrupted; recorded the active run status.")
        except Exception as error:
            batch_error = compact_error(f"{error.__class__.__name__}: {error}")
            self.progress.write(f"Batch setup failed: {batch_error}")
        finally:
            if self.args.dry_run:
                batch_status = "dry_run"
            elif interrupted:
                batch_status = "interrupted"
            elif batch_error:
                batch_status = "failure"
            elif self.failed_configurations:
                batch_status = "partial_failure"
            else:
                batch_status = "success"
            self.recorder.write(
                self._batch_record(
                    "batch_finish",
                    status=batch_status,
                    started_at=batch_started_at,
                    finished_at=now_iso(),
                    duration_seconds=round(
                        time.monotonic() - batch_started_monotonic, 3
                    ),
                    completed_configuration_count=self.completed_configurations,
                    failed_configuration_count=self.failed_configurations,
                    error=batch_error,
                )
            )
            self.progress.close()
            self.recorder.close()

        self.progress.write(
            f"Finished {self.args.matrix_name}: status={batch_status}; "
            f"status index={self.status_path}"
        )
        if interrupted:
            return 130
        return 1 if batch_status in {"failure", "partial_failure", "interrupted"} else 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a teacher prompt matrix with tqdm progress and JSONL status."
    )
    parser.add_argument("--matrix-name", required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        choices=range(1, len(MATRIX_BACKEND_CONTAINERS) + 1),
        default=DEFAULT_CONFIGURATION_WORKERS,
        metavar=f"1-{len(MATRIX_BACKEND_CONTAINERS)}",
        help=(
            "Number of isolated configuration workers to run concurrently "
            f"(default: {DEFAULT_CONFIGURATION_WORKERS})."
        ),
    )
    parser.add_argument(
        "--setup-jobs",
        type=int,
        choices=range(1, len(MATRIX_BACKEND_CONTAINERS) + 1),
        default=DEFAULT_REPO_SETUP_WORKERS,
        metavar=f"1-{len(MATRIX_BACKEND_CONTAINERS)}",
        help=(
            "Maximum repository setup/build scripts allowed concurrently "
            f"(default: {DEFAULT_REPO_SETUP_WORKERS})."
        ),
    )
    parser.add_argument(
        "--backend-container",
        action="append",
        default=[],
        help=(
            "Explicit isolated backend container to use; repeat once per job. "
            "The launcher uses this to create the shared worker pool."
        ),
    )
    parser.add_argument(
        "--backend-log-root",
        action="append",
        default=[],
        type=parse_backend_log_root,
        metavar="CONTAINER=PATH",
        help="Host artifact root mounted into one backend container.",
    )
    parser.add_argument(
        "--teacher-mode",
        required=True,
        choices=("observe", "steer", "objective_rewrite"),
    )
    parser.add_argument(
        "--prompt-placement",
        choices=tuple(PROMPT_PLACEMENT_MAP),
        default="both",
        help=(
            "Run custom prompts as a user-message prefix, a Gemini system "
            "instruction, or both variants separately."
        ),
    )
    parser.add_argument(
        "--teacher-max-input-tokens",
        type=positive_int,
        default=TEACHER_MAX_INPUT_TOKENS,
        help=(
            "Maximum Gemini teacher input context "
            f"(default: {TEACHER_MAX_INPUT_TOKENS})."
        ),
    )
    parser.add_argument(
        "--teacher-max-output-tokens",
        type=positive_int,
        default=TEACHER_MAX_OUTPUT_TOKENS,
        help=(
            "Maximum Gemini teacher response length "
            f"(default: {TEACHER_MAX_OUTPUT_TOKENS})."
        ),
    )
    parser.add_argument(
        "--phase-iterations",
        type=positive_int,
        default=DEFAULT_PHASE_ITERATIONS,
        help=(
            "Maximum student iterations in each workflow phase "
            f"(default: {DEFAULT_PHASE_ITERATIONS})."
        ),
    )
    parser.add_argument("--prompt-file", action="append", required=True)
    parser.add_argument(
        "--exclude-baseline",
        action="store_true",
        help="Do not add teacher-active runs without a custom teacher prompt.",
    )
    parser.add_argument(
        "--environment", action="append", type=parse_environment, required=True
    )
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--launcher-argument", action="append", default=[])
    parser.add_argument(
        "--skip-configurations-from",
        action="append",
        default=[],
        metavar="STATUS_JSONL",
        help=(
            "May be repeated; skip finalized top-level configurations recorded "
            "as successful in a prior run_status.jsonl."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prune-dind-between-repositories",
        action="store_true",
        help=(
            "After each environment finishes, prune unused containers, networks, "
            "volumes, dangling images, and build cache inside the backend's Docker "
            "daemon. Tagged images are preserved."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print every child workflow log line to the terminal.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm rendering for redirected output or CI.",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if len(args.environment) != 9:
        raise SystemExit(
            f"Expected exactly 9 environments, found {len(args.environment)}"
        )
    if not 1 <= len(args.prompt_file) <= 5:
        raise SystemExit(f"Expected 1 through 5 prompts, found {len(args.prompt_file)}")
    return MatrixRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
