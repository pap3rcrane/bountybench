#!/usr/bin/env python3
"""Download and organize matrix trajectories from one GCP VM.

The VM is read-only: files are streamed through stdout and nothing is created,
deleted, or modified remotely. The organized files are written locally under:

  ~/Desktop/bountybench-downloads/<environment>/
    <environment>__<placement>__<category>__<prompt>/
      run1.json
      run2.json

Task Designer configurations preserve all three source trajectories as
``run1_source_1.json``, ``run1_source_2.json``, and ``run1_source_3.json``, plus
the complete teacher episode as ``run1_teacher_trace.json``. Only the sanitized
JSON artifacts are retained; run directories, full logs, batch logs, and the
separate generated-objective artifact are not retained.

Example (checks all eight experiment VMs):

  python scripts/download_vm_matrix_traces.py

Pass one or more VM names to limit the download:

  python scripts/download_vm_matrix_traces.py cyber-seed-prompts-4b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from messages.trajectory_sanitizer import sanitize_trajectory


DEFAULT_PROJECT = "gdm-finetuning"
DEFAULT_VMS = (
    "cyber-seed-prompts-3a",
    "cyber-seed-prompts-4a",
    "cyber-seed-prompts-3b",
    "cyber-seed-prompts-4b",
    "cyber-seed-prompts-5",
    "cyber-seed-prompts-6",
    "cyber-seed-prompts-7",
    "cyber-seed-prompts-8",
)
VM_ZONES = {
    "cyber-seed-prompts-3a": "us-east5-a",
    "cyber-seed-prompts-4a": "us-east1-b",
    "cyber-seed-prompts-3b": "us-east5-a",
    "cyber-seed-prompts-4b": "us-east5-a",
    "cyber-seed-prompts-5": "us-central1-a",
    "cyber-seed-prompts-6": "us-east1-c",
    "cyber-seed-prompts-7": "us-east1-c",
    "cyber-seed-prompts-8": "us-east4-b",
}
CATEGORY_NAMES = {
    "observe": "critiquer",
    "steer": "judge_verifier",
    "objective_rewrite": "task_designer",
}
PLACEMENT_NAMES = {
    "prepend": "user",
    "user": "user",
    "system": "system",
    "none": "none",
}
SELECTED_ENVIRONMENTS = {
    # Twenty-five environments with the strongest available successful coverage.
    # InvokeAI is absent because it has no downloadable successes, and llama_index
    # is absent because it has only 15. Node is retained as the 25th environment
    # because it has the full 55 successful Task Designer configurations.
    ("observe", "LibreChat", "0", "exploit_workflow"),
    ("observe", "agentscope", "0", "exploit_workflow"),
    ("observe", "astropy", "0", "detect_workflow"),
    ("observe", "django", "0", "detect_workflow"),
    ("observe", "gpt_academic", "0", "detect_workflow"),
    ("observe", "kedro", "0", "patch_workflow"),
    ("observe", "scikit-learn", "0", "patch_workflow"),
    ("observe", "yaml", "0", "patch_workflow"),
    ("steer", "mlflow", "0", "exploit_workflow"),
    ("steer", "lunary", "0", "exploit_workflow"),
    ("steer", "fastapi", "0", "exploit_workflow"),
    ("steer", "curl", "0", "detect_workflow"),
    ("steer", "gluon-cv", "0", "detect_workflow"),
    ("steer", "gunicorn", "0", "detect_workflow"),
    ("steer", "setuptools", "0", "patch_workflow"),
    ("steer", "zipp", "0", "patch_workflow"),
    ("objective_rewrite", "gradio", "0", "exploit_workflow"),
    ("objective_rewrite", "composio", "0", "exploit_workflow"),
    ("objective_rewrite", "node", "0", "exploit_workflow"),
    ("objective_rewrite", "bentoml", "0", "detect_workflow"),
    ("objective_rewrite", "langchain", "0", "detect_workflow"),
    ("objective_rewrite", "pytorch-lightning", "0", "detect_workflow"),
    ("objective_rewrite", "paddle", "0", "patch_workflow"),
    ("objective_rewrite", "parse-url", "0", "patch_workflow"),
    ("objective_rewrite", "undici", "0", "patch_workflow"),
}
OUTPUT_LAYOUT = "flat_trajectory_files_v3"
OBJECTIVE_INDEX_FILE = ".objective_index.json"


@dataclass
class Attempt:
    """One execution attempt for one final matrix configuration."""

    vm: str
    staging_root: Path
    status_file: Path
    batch: dict[str, Any]
    configuration_id: str
    configuration: dict[str, Any] | None = None
    student_runs: list[dict[str, Any]] = field(default_factory=list)
    teacher_objective: Path | None = None
    teacher_trace: Path | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self.configuration:
            return self.configuration
        if self.student_runs:
            return self.student_runs[0]
        return {}

    @property
    def logical_key(self) -> tuple[str, ...]:
        record = self.metadata
        return (
            str(record.get("repo_name") or "unknown_repo"),
            str(record.get("bounty_number") or "unknown"),
            str(record.get("workflow_type") or "unknown_workflow"),
            str(
                record.get("teacher_type")
                or self.batch.get("teacher_type")
                or "unknown"
            ),
            str(record.get("system_prompt_name") or "none"),
            str(record.get("system_prompt_placement") or "none"),
            str(record.get("run_number") or "unknown"),
        )

    @property
    def finished_at(self) -> str:
        records = [r for r in [self.configuration, *self.student_runs] if r]
        return max((str(r.get("finished_at") or "") for r in records), default="")

    def rank(self) -> tuple[int, float, int, int, int]:
        """Prefer an effective success, then the newest qualifying attempt."""
        trace_records = [
            record
            for record in self.student_runs
            if resolve_downloaded_path(
                record.get("workflow_log_path"), self.staging_root
            )
        ]
        successful_traces = sum(r.get("status") == "success" for r in trace_records)
        configuration_success = int(
            bool(self.configuration and self.configuration.get("status") == "success")
        )
        return (
            int(is_effective_success(self)),
            timestamp_value(self.finished_at),
            configuration_success,
            successful_traces,
            len(trace_records),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read matrix statuses and trajectories from a GCP VM, then organize "
            "them on this local machine."
        )
    )
    parser.add_argument(
        "vms",
        nargs="*",
        help=(
            "Optional GCP VM names. With none supplied, checks "
            "cyber-seed-prompts-3a through 8."
        ),
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP project ID")
    parser.add_argument(
        "--zone",
        help="Zone override, allowed only when downloading one VM.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.home() / "Desktop" / "bountybench-downloads",
        help="Local parent directory (default: ~/Desktop/bountybench-downloads)",
    )
    parser.add_argument(
        "--remote-repo",
        default="$HOME/bountybench",
        help="Repository path on the VM (default: $HOME/bountybench)",
    )
    parser.add_argument(
        "--from-local-export",
        type=Path,
        help=(
            "Skip GCP and organize an already extracted export containing runs/ "
            "and batch_logs/. Useful for testing or re-organizing a prior download."
        ),
    )
    return parser.parse_args()


def stream_export_from_vm(
    *,
    vm: str,
    project: str,
    zone: str | None,
    remote_repo: str,
    staging_root: Path,
) -> None:
    """Stream status metadata and trajectory JSON needed locally."""
    # All remote operations below are read-only. tar writes its archive to stdout.
    remote_command = rf"""set -eu
cd {remote_repo}
{{
  find runs -type f -name run_status.jsonl -print0 2>/dev/null || true
  find batch_logs -type f -path '*/logs/*.json' -print0 2>/dev/null || true
  find logs -type f -name '*.json' -print0 2>/dev/null || true
  find batch_logs -type f -path '*/generated_objectives/*_objective.json' -print0 2>/dev/null || true
  find batch_logs -type f -path '*/generated_objectives/*_teacher_trace.json' -print0 2>/dev/null || true
  find generated_objectives -type f -name '*_objective.json' -print0 2>/dev/null || true
  find generated_objectives -type f -name '*_teacher_trace.json' -print0 2>/dev/null || true
  find batch_logs -mindepth 3 -maxdepth 3 -type f -name '*.log' -exec grep -lZE -m1 'Wrote rewritten objective:|TEACHER RESPONSE \(objective_rewrite\)' {{}} + 2>/dev/null || true
}} | tar --ignore-failed-read --warning=no-file-changed --null --files-from=- --create --gzip --file=-
"""
    ssh_command = vm_ssh_command(
        vm=vm,
        project=project,
        zone=zone,
        remote_command=remote_command,
    )
    extract_command = ["tar", "-xzf", "-", "-C", str(staging_root)]

    zone_label = f" ({zone})" if zone else ""
    print(f"Streaming statuses and trajectories from {vm}{zone_label}...")
    remote = subprocess.Popen(ssh_command, stdout=subprocess.PIPE)
    assert remote.stdout is not None
    try:
        extracted = subprocess.run(extract_command, stdin=remote.stdout, check=False)
    finally:
        remote.stdout.close()
    remote_code = remote.wait()
    if remote_code != 0 or extracted.returncode != 0:
        raise RuntimeError(
            "Download failed: "
            f"gcloud exit={remote_code}, local tar exit={extracted.returncode}."
        )
    status_count = sum(1 for _ in staging_root.glob("runs/**/run_status.jsonl"))
    artifact_count = sum(
        1 for _ in staging_root.glob("batch_logs/matrix_workers/**/logs/**/*.json")
    ) + sum(1 for _ in staging_root.glob("logs/**/*.json"))
    print(
        f"Downloaded from {vm}: {status_count} status files, "
        f"{artifact_count} trajectories."
    )
    if status_count and artifact_count == 0:
        raise RuntimeError(
            f"{vm} supplied status files but no trajectories; refusing to create "
            "an empty organized result."
        )


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOGGER_HEADER = re.compile(
    r"^\d{4}-\d{2}-\d{2} .*\[[^\]]+\.py:\d+\]$"
)


def _clean_log_lines(path: Path) -> list[str]:
    return [
        ANSI_ESCAPE.sub("", line).rstrip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]


def _logged_section(lines: list[str], marker_index: int) -> str:
    end = len(lines)
    for index in range(marker_index + 1, len(lines)):
        if LOGGER_HEADER.match(lines[index]):
            end = index
            break
    return "\n".join(lines[marker_index + 1 : end]).strip()


def _reported_objective_name(lines: list[str]) -> str | None:
    marker = "Wrote rewritten objective:"
    for index, line in enumerate(lines):
        if marker not in line:
            continue
        reported = line.split(marker, 1)[1].strip()
        for continuation in lines[index + 1 : index + 5]:
            if "_objective.json" in reported:
                break
            reported += continuation.strip()
        if "_objective.json" in reported:
            return Path(reported).name
    return None


def _reconstruct_historical_teacher_trace(
    log_path: Path, objective_path: Path | None = None
) -> dict[str, Any] | None:
    """Recover the actual offline teacher request/response from its batch log."""
    lines = _clean_log_lines(log_path)
    teacher_markers = [
        index
        for index, line in enumerate(lines)
        if "TEACHER RESPONSE (objective_rewrite)" in line
    ]
    if not teacher_markers:
        return None
    teacher_index = teacher_markers[-1]
    input_markers = [
        index
        for index, line in enumerate(lines[:teacher_index])
        if "Model input (truncated if over max tokens):" in line
    ]
    response_markers = [
        index
        for index, line in enumerate(lines[:teacher_index])
        if line == "Unparsed LM Response:"
    ]
    if not input_markers or not response_markers:
        return None
    input_index = input_markers[-1]
    raw_response_index = response_markers[-1]
    previous_response = max(
        (index for index in response_markers if index < input_index), default=-1
    )
    system_markers = [
        index
        for index, line in enumerate(lines[previous_response + 1 : input_index])
        if line == "Model system prompt:"
    ]
    system_prompt = None
    if system_markers:
        system_index = previous_response + 1 + system_markers[-1]
        system_prompt = _logged_section(lines, system_index)

    raw_section = _logged_section(lines, raw_response_index)
    if raw_section.startswith("content:\n"):
        raw_section = raw_section[len("content:\n") :]
    usage_boundary = raw_section.rfind("\ninput_tokens:\n")
    if usage_boundary >= 0:
        raw_section = raw_section[:usage_boundary].rstrip()
    if not raw_section:
        return None

    if objective_path is not None:
        try:
            json.loads(objective_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return {
        "teacher_mode": "objective_rewrite",
        "system_prompt": system_prompt,
        "input": _logged_section(lines, input_index),
        "raw_response": raw_section,
    }


def build_objective_index(staging_root: Path) -> None:
    """Index objectives and create historical teacher traces locally."""
    objectives: dict[str, list[Path]] = {}
    traces: dict[str, list[Path]] = {}
    for pattern, destination in (
        ("batch_logs/**/generated_objectives/*_objective.json", objectives),
        ("generated_objectives/*_objective.json", objectives),
        ("batch_logs/**/generated_objectives/*_teacher_trace.json", traces),
        ("generated_objectives/*_teacher_trace.json", traces),
    ):
        for path in staging_root.glob(pattern):
            destination.setdefault(path.name, []).append(path)

    reconstructed_root = staging_root / ".reconstructed_teacher_traces"
    index: dict[str, dict[str, str]] = {}
    for log_path in staging_root.glob("batch_logs/*/*/*.log"):
        lines = _clean_log_lines(log_path)
        objective_name = _reported_objective_name(lines)
        matches = objectives.get(objective_name or "", [])
        objective_path = matches[0] if len(matches) == 1 else None
        trace_path: Path | None = None
        if objective_path is not None:
            trace_name = objective_path.name.replace(
                "_objective.json", "_teacher_trace.json"
            )
            trace_matches = traces.get(trace_name, [])
            if len(trace_matches) == 1:
                trace_path = trace_matches[0]
        if trace_path is None:
            teacher_trace = _reconstruct_historical_teacher_trace(
                log_path, objective_path
            )
            if teacher_trace is None:
                continue
            reconstructed_root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(
                str(log_path.relative_to(staging_root)).encode()
            ).hexdigest()[:16]
            trace_path = reconstructed_root / f"{digest}_teacher_trace.json"
            trace_path.write_text(
                json.dumps(teacher_trace, indent=2) + "\n", encoding="utf-8"
            )
        entry = {
            "teacher_trace_path": str(trace_path.relative_to(staging_root)),
        }
        if objective_path is not None:
            entry["objective_path"] = str(objective_path.relative_to(staging_root))
        index[str(log_path.relative_to(staging_root))] = entry
    (staging_root / OBJECTIVE_INDEX_FILE).write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )


def vm_ssh_command(
    *, vm: str, project: str, zone: str | None, remote_command: str
) -> list[str]:
    command = [
        "gcloud",
        "compute",
        "ssh",
        vm,
        f"--project={project}",
        "--quiet",
        f"--command={remote_command}",
    ]
    if zone:
        command.insert(5, f"--zone={zone}")
    return command


def snapshot_vm_files(
    *, vm: str, project: str, zone: str | None, remote_repo: str
) -> dict[str, tuple[str, str]]:
    """Read size and high-resolution mtime without changing remote files."""
    remote_command = rf"""set -eu
cd {remote_repo}
{{
  find runs -type f -name run_status.jsonl -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find batch_logs -type f -path '*/logs/*.json' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find logs -type f -name '*.json' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find batch_logs -type f -path '*/generated_objectives/*_objective.json' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find batch_logs -type f -path '*/generated_objectives/*_teacher_trace.json' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find generated_objectives -type f -name '*_objective.json' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find generated_objectives -type f -name '*_teacher_trace.json' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
  find batch_logs -mindepth 3 -maxdepth 3 -type f -name '*.log' -printf '%p\0%s\0%T@\0' 2>/dev/null || true
}}
"""
    result = subprocess.run(
        vm_ssh_command(
            vm=vm,
            project=project,
            zone=zone,
            remote_command=remote_command,
        ),
        check=True,
        stdout=subprocess.PIPE,
    )
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        raise RuntimeError(f"Could not parse the file snapshot returned by {vm}.")
    snapshot: dict[str, tuple[str, str]] = {}
    for index in range(0, len(fields), 3):
        path = fields[index].decode("utf-8", errors="surrogateescape")
        size = fields[index + 1].decode("ascii", errors="replace")
        modified = fields[index + 2].decode("ascii", errors="replace")
        snapshot[path] = (size, modified)
    return snapshot


def discard_unstable_files(
    *,
    staging_root: Path,
    before: dict[str, tuple[str, str]],
    after: dict[str, tuple[str, str]],
) -> set[str]:
    """Remove temporary copies whose remote source changed during transfer."""
    unstable = {
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }
    for relative_path in unstable:
        candidate = staging_root / relative_path
        if candidate.is_file():
            candidate.unlink()
    return unstable


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A run may still be appending its last JSONL record while downloaded.
                print(
                    f"Warning: ignored incomplete JSONL line {line_number} in {path}",
                    file=sys.stderr,
                )
                continue
            if isinstance(value, dict):
                yield value


def collect_attempts(staging_root: Path, vm: str) -> list[Attempt]:
    objective_index_path = staging_root / OBJECTIVE_INDEX_FILE
    if not objective_index_path.is_file():
        build_objective_index(staging_root)
    try:
        objective_index = json.loads(objective_index_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        objective_index = {}
    if not isinstance(objective_index, dict):
        objective_index = {}

    attempts: list[Attempt] = []
    for status_file in sorted(staging_root.glob("runs/**/run_status.jsonl")):
        records = list(read_json_lines(status_file))
        batch = next(
            (
                record
                for record in records
                if record.get("record_type") == "batch_start"
            ),
            {},
        )
        grouped: dict[str, Attempt] = {}
        for record in records:
            if record.get("record_type") not in {"configuration", "student_run"}:
                continue
            configuration_id = record.get("configuration_id")
            if not configuration_id:
                continue
            attempt = grouped.setdefault(
                str(configuration_id),
                Attempt(
                    vm=vm,
                    staging_root=staging_root,
                    status_file=status_file,
                    batch=batch,
                    configuration_id=str(configuration_id),
                ),
            )
            if record.get("record_type") == "configuration":
                attempt.configuration = record
            else:
                attempt.student_runs.append(record)
        for attempt in grouped.values():
            if attempt.configuration:
                attempt.teacher_objective = resolve_downloaded_path(
                    attempt.configuration.get("teacher_objective_path"),
                    staging_root,
                )
                attempt.teacher_trace = resolve_downloaded_path(
                    attempt.configuration.get("teacher_trace_path"),
                    staging_root,
                )
            if attempt.teacher_trace:
                continue
            configuration_logs = {
                str(record.get("configuration_log_path") or "")
                for record in [attempt.configuration, *attempt.student_runs]
                if record
            }
            for configuration_log in configuration_logs:
                normalized = configuration_log.replace("\\", "/")
                marker_position = normalized.find("batch_logs/")
                if marker_position >= 0:
                    normalized = normalized[marker_position:]
                objective_entry = objective_index.get(normalized)
                if not isinstance(objective_entry, dict):
                    continue
                attempt.teacher_objective = resolve_downloaded_path(
                    objective_entry.get("objective_path"), staging_root
                )
                trace_value = objective_entry.get("teacher_trace_path")
                if trace_value:
                    trace_candidate = staging_root / str(trace_value)
                    if trace_candidate.is_file():
                        attempt.teacher_trace = trace_candidate
                if attempt.teacher_trace:
                    break
                attempt.teacher_trace = None
        attempts.extend(grouped.values())
    return attempts


def timestamp_value(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def path_from_repo_marker(path: str, staging_root: Path) -> Path | None:
    normalized = path.replace("\\", "/")
    for marker in (
        "batch_logs/",
        "logs/",
        "full_logs/",
        "generated_objectives/",
        "runs/",
    ):
        position = normalized.find(marker)
        if position >= 0:
            candidate = staging_root / normalized[position:]
            if candidate.is_file():
                return candidate
    return None


def resolve_downloaded_path(value: Any, staging_root: Path) -> Path | None:
    if not value:
        return None
    return path_from_repo_marker(str(value), staging_root)


def sanitize(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "unknown"


def write_sanitized_trajectory(source: Path, destination: Path) -> None:
    trajectory = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(
        json.dumps(
            sanitize_trajectory(trajectory, expose_observe_feedback=True),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_teacher_trace_artifact(
    source: Path, destination: Path, attempt: Attempt
) -> None:
    trace = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(trace, dict):
        raise ValueError(f"Teacher trace must be a JSON object: {source}")
    repository, bounty, workflow, teacher_type, _, placement, repeat = (
        attempt.logical_key
    )
    trace.setdefault("teacher_mode", teacher_type)
    trace.setdefault(
        "model",
        attempt.metadata.get("teacher_model")
        or attempt.batch.get("teacher_model"),
    )
    trace.setdefault("system_prompt_placement", placement)
    trace.setdefault(
        "configuration",
        {
            "repository": repository,
            "bounty_number": bounty,
            "workflow_type": workflow,
            "run_number": repeat,
        },
    )
    for redundant_field in ("response", "objective", "record_source"):
        trace.pop(redundant_field, None)
    destination.write_text(
        json.dumps(
            sanitize_trajectory(trace, expose_observe_feedback=False), indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def output_parts(attempt: Attempt) -> tuple[str, str, int]:
    (
        repository,
        bounty_number,
        workflow_type,
        teacher_type,
        prompt_name,
        placement,
        run_number,
    ) = attempt.logical_key
    environment = sanitize(f"{repository}_bounty_{bounty_number}_{workflow_type}")
    placement_name = sanitize(PLACEMENT_NAMES.get(placement, placement))
    category = sanitize(CATEGORY_NAMES.get(teacher_type, teacher_type))
    subtype = sanitize(
        "no_custom_prompt" if prompt_name in {"", "none", "null"} else prompt_name
    )
    configuration = f"{environment}__{placement_name}__{category}__{subtype}"
    try:
        repeat = int(run_number)
    except ValueError as error:
        raise RuntimeError(
            f"Invalid run number {run_number!r} in {attempt.status_file}"
        ) from error
    return environment, configuration, repeat


def selected_trace_records(attempt: Attempt) -> list[tuple[dict[str, Any], Path]]:
    records: list[tuple[dict[str, Any], Path]] = []
    for record in attempt.student_runs:
        path = resolve_downloaded_path(
            record.get("workflow_log_path"), attempt.staging_root
        )
        if path:
            records.append((record, path))
    records.sort(
        key=lambda item: (
            item[0].get("status") == "success",
            item[0].get("run_role") == "normal",
            str(item[0].get("finished_at") or ""),
        ),
        reverse=True,
    )
    return records


def is_effective_success(attempt: Attempt) -> bool:
    if attempt.configuration and attempt.configuration.get("status") == "success":
        return True
    return any(
        record.get("status") == "success"
        and resolve_downloaded_path(
            record.get("workflow_log_path"), attempt.staging_root
        )
        for record in attempt.student_runs
    )


def is_selected_logical_key(logical_key: tuple[str, ...]) -> bool:
    repository, bounty_number, workflow_type, teacher_type = logical_key[:4]
    return (
        teacher_type,
        repository,
        bounty_number,
        workflow_type,
    ) in SELECTED_ENVIRONMENTS


def complete_trajectories(
    attempt: Attempt,
) -> tuple[bool, str | None, list[tuple[dict[str, Any], Path]]]:
    """Require every trajectory expected for one usable configuration."""
    traces = [
        (record, path)
        for record, path in selected_trace_records(attempt)
        if record.get("status") == "success"
    ]

    if attempt.logical_key[3] == "objective_rewrite":
        source_traces = {
            str(record.get("run_role")): (record, trace_path)
            for record, trace_path in traces
            if record.get("run_role") in {"source_1", "source_2", "source_3"}
        }
        if set(source_traces) != {"source_1", "source_2", "source_3"}:
            return (
                False,
                "one or more source trajectories are missing or changed during download",
                traces,
            )
        if attempt.teacher_trace is None:
            return (
                False,
                "nonempty teacher output is missing or changed during download",
                traces,
            )
        return True, None, [source_traces[role] for role in sorted(source_traces)]

    normal_traces = [
        (record, trace_path)
        for record, trace_path in traces
        if record.get("run_role") == "normal"
    ]
    if len(normal_traces) != 1:
        return (
            False,
            "normal student trajectory is missing or changed during download",
            traces,
        )
    return True, None, normal_traces


def trajectory_outputs(
    *,
    repeat: int,
    attempt: Attempt,
    traces: list[tuple[dict[str, Any], Path]],
) -> list[tuple[str, str, Path]]:
    """Return manifest key, flat output name, and source for each trajectory."""
    objective_rewrite = attempt.logical_key[3] == "objective_rewrite"
    outputs: list[tuple[str, str, Path]] = []
    for record, trace_path in traces:
        role = sanitize(record.get("run_role") or "unknown")
        if not objective_rewrite and role != "normal":
            continue
        if objective_rewrite:
            outputs.append(
                (f"{role}_trajectory", f"run{repeat}_{role}.json", trace_path)
            )
        else:
            outputs.append(("trajectory", f"run{repeat}.json", trace_path))
    if objective_rewrite and attempt.teacher_trace is not None:
        outputs.append(
            (
                "teacher_trace",
                f"run{repeat}_teacher_trace.json",
                attempt.teacher_trace,
            )
        )
    return outputs


def install_trajectories_atomically(
    *,
    configuration_dir: Path,
    repeat: int,
    attempt: Attempt,
    traces: list[tuple[dict[str, Any], Path]],
) -> dict[str, str]:
    """Replace one run with flat trajectory files and remove its old folder."""
    configuration_dir.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".run{repeat}-download-", dir=configuration_dir)
    )
    payload = temporary_parent / "payload"
    backup = temporary_parent / "previous"
    payload.mkdir()
    backup.mkdir()
    outputs = trajectory_outputs(repeat=repeat, attempt=attempt, traces=traces)
    targets = {key: (configuration_dir / name, source) for key, name, source in outputs}
    legacy_run_dir = configuration_dir / f"run{repeat}"
    obsolete_objective = configuration_dir / f"run{repeat}_teacher_objective.json"
    moved_to_backup: list[tuple[Path, Path]] = []
    try:
        for key, (target, source) in targets.items():
            if key == "teacher_trace":
                write_teacher_trace_artifact(
                    source, payload / target.name, attempt
                )
            else:
                write_sanitized_trajectory(source, payload / target.name)
        for target in [
            *(target for target, _ in targets.values()),
            legacy_run_dir,
            obsolete_objective,
        ]:
            if target.exists():
                backup_path = backup / target.name
                target.replace(backup_path)
                moved_to_backup.append((backup_path, target))
        for target, _ in targets.values():
            (payload / target.name).replace(target)
        return {key: str(target) for key, (target, _) in targets.items()}
    except Exception:
        for target, _ in targets.values():
            if target.exists():
                target.unlink()
        for backup_path, target in reversed(moved_to_backup):
            if backup_path.exists():
                backup_path.replace(target)
        raise
    finally:
        if temporary_parent.exists():
            shutil.rmtree(temporary_parent)


def read_existing_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "download_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_manifest_atomically(output_root: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = output_root / "download_manifest.json"
    temporary_path = output_root / ".download_manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path


def remove_record_artifacts(record: dict[str, Any], output_root: Path) -> None:
    """Remove files for a manifest record only when they are under output_root."""
    candidates = [Path(path) for path in (record.get("files") or {}).values()]
    run_directory = record.get("run_directory")
    if run_directory:
        candidates.append(Path(run_directory))

    removed_parents: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
            resolved.relative_to(output_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            resolved.unlink()
            removed_parents.add(resolved.parent)
        elif resolved.is_dir():
            shutil.rmtree(resolved)
            removed_parents.add(resolved.parent)

    for parent in sorted(
        removed_parents, key=lambda path: len(path.parts), reverse=True
    ):
        while parent != output_root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def merge_vm_attempts(
    attempts: list[Attempt],
    destination_root: Path,
    completed_vm: str,
    unstable_files: set[str] | None = None,
) -> dict[str, Any]:
    """Merge one VM immediately while preserving newer local successes."""
    output_root = destination_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing = read_existing_manifest(output_root)
    records_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    excluded_existing_records = 0
    for record in existing.get("records", []):
        logical_key = record.get("logical_key")
        files = record.get("files") or {}
        key = (
            tuple(str(part) for part in logical_key)
            if isinstance(logical_key, list)
            else ()
        )
        if key and not is_selected_logical_key(key):
            remove_record_artifacts(record, output_root)
            excluded_existing_records += 1
            continue
        if (
            key
            and record.get("effective_success")
            and files
            and all(Path(path).is_file() for path in files.values())
        ):
            records_by_key[key] = record

    skipped_updates = list(existing.get("skipped_unstable_or_incomplete_updates", []))
    candidates_on_vm: dict[
        tuple[str, ...],
        list[tuple[Attempt, list[tuple[dict[str, Any], Path]]]],
    ] = {}
    for attempt in attempts:
        if not is_selected_logical_key(attempt.logical_key) or not is_effective_success(
            attempt
        ):
            continue
        complete, reason, traces = complete_trajectories(attempt)
        if not complete:
            skipped_updates.append(
                {
                    "vm": attempt.vm,
                    "logical_key": list(attempt.logical_key),
                    "finished_at": attempt.finished_at,
                    "reason": reason,
                }
            )
            continue
        candidates_on_vm.setdefault(attempt.logical_key, []).append((attempt, traces))

    # Keep one older usable candidate behind the newest candidate. It is held
    # only during the merge and is never emitted as an additional output file.
    for key, candidates in candidates_on_vm.items():
        candidates_on_vm[key] = sorted(
            candidates,
            key=lambda candidate: candidate[0].rank(),
            reverse=True,
        )[:2]

    updates = 0
    fallback_updates = 0
    for key, candidates in sorted(candidates_on_vm.items()):
        current = records_by_key.get(key)
        current_rank = tuple(current.get("selection_rank", [])) if current else ()
        for candidate_index, (attempt, traces) in enumerate(candidates):
            if (
                current
                and current.get("layout") == OUTPUT_LAYOUT
                and tuple(attempt.rank()) <= current_rank
            ):
                break

            environment, configuration, repeat = output_parts(attempt)
            configuration_dir = output_root / environment / configuration
            try:
                files = install_trajectories_atomically(
                    configuration_dir=configuration_dir,
                    repeat=repeat,
                    attempt=attempt,
                    traces=traces,
                )
            except (json.JSONDecodeError, OSError, UnicodeError) as error:
                skipped_updates.append(
                    {
                        "vm": attempt.vm,
                        "logical_key": list(attempt.logical_key),
                        "finished_at": attempt.finished_at,
                        "reason": (
                            "candidate could not be sanitized or installed; "
                            f"kept prior version: {error}"
                        ),
                    }
                )
                continue

            records_by_key[key] = {
                "logical_key": list(key),
                "environment": environment,
                "configuration_folder": configuration,
                "run_number": repeat,
                "layout": OUTPUT_LAYOUT,
                "configuration_directory": str(configuration_dir),
                "files": files,
                "source_vm": attempt.vm,
                "selected_batch_id": attempt.batch.get("batch_id"),
                "selected_configuration_id": attempt.configuration_id,
                "selected_finished_at": attempt.finished_at,
                "selection_rank": list(attempt.rank()),
                "configuration_status": (
                    attempt.configuration.get("status")
                    if attempt.configuration
                    else None
                ),
                "effective_success": True,
                "student_statuses": {
                    str(record.get("run_role") or "unknown"): record.get("status")
                    for record in attempt.student_runs
                },
                "source_status_file": (
                    f"{attempt.vm}:"
                    f"{attempt.status_file.relative_to(attempt.staging_root)}"
                ),
                "used_older_backup": candidate_index == 1,
            }
            updates += 1
            fallback_updates += int(candidate_index == 1)
            break

    checked_vms = list(existing.get("checked_vms", []))
    if completed_vm not in checked_vms:
        checked_vms.append(completed_vm)
    unstable_by_vm = dict(existing.get("unstable_files_by_vm", {}))
    unstable_by_vm[completed_vm] = {
        "count": len(unstable_files or ()),
        "paths": sorted(unstable_files or ()),
    }
    manifest = {
        "checked_vms": checked_vms,
        "created_at": existing.get("created_at")
        or datetime.now().astimezone().isoformat(),
        "updated_at": datetime.now().astimezone().isoformat(),
        "output_layout": OUTPUT_LAYOUT,
        "selection_rule": (
            "Search every run status directory on every checked VM, then keep the "
            "most recent complete successful trajectory for each configuration in "
            "the fixed 25-environment selection. A Task Designer configuration is "
            "complete when all three source trajectories succeeded and its batch "
            "log contains nonempty teacher output; strict objective parsing is not "
            "required. Keep one older usable candidate during each merge. Files "
            "changed during transfer or candidates that cannot be sanitized never "
            "replace a local version; use the older candidate when available."
        ),
        "selected_environment_count": len(SELECTED_ENVIRONMENTS),
        "selected_environments": [
            {
                "teacher_type": teacher_type,
                "repo_name": repository,
                "bounty_number": bounty_number,
                "workflow_type": workflow_type,
            }
            for teacher_type, repository, bounty_number, workflow_type in sorted(
                SELECTED_ENVIRONMENTS
            )
        ],
        "excluded_existing_records_removed": excluded_existing_records,
        "effective_success_count": len(records_by_key),
        "trajectory_runs_written": len(records_by_key),
        "updates_saved_from_last_vm": updates,
        "older_backups_used_from_last_vm": fallback_updates,
        "last_completed_vm": completed_vm,
        "unstable_files_by_vm": unstable_by_vm,
        "skipped_unstable_or_incomplete_updates": skipped_updates,
        "records": [records_by_key[key] for key in sorted(records_by_key)],
    }
    manifest_path = write_manifest_atomically(output_root, manifest)
    return {**manifest, "root": str(output_root), "manifest": str(manifest_path)}


def main() -> int:
    args = parse_args()
    vms = list(dict.fromkeys(args.vms or DEFAULT_VMS))
    if args.zone and len(vms) != 1:
        raise RuntimeError("--zone can only be used when exactly one VM is selected.")
    if args.from_local_export:
        if len(vms) != 1:
            raise RuntimeError(
                "--from-local-export requires exactly one explicit VM label."
            )
        staging_root = args.from_local_export.expanduser().resolve()
        if not staging_root.is_dir():
            raise RuntimeError(f"Local export does not exist: {staging_root}")
        attempts = collect_attempts(staging_root, vms[0])
        result = merge_vm_attempts(attempts, args.destination, vms[0], set())
    else:
        result: dict[str, Any] = {}
        for vm in vms:
            zone = args.zone or VM_ZONES.get(vm)
            if not zone:
                raise RuntimeError(
                    f"No saved zone for {vm!r}. Pass --zone when downloading "
                    "a VM not listed in VM_ZONES."
                )
            with tempfile.TemporaryDirectory(prefix=f"{sanitize(vm)}-matrix-") as temp:
                staging_root = Path(temp)
                print(f"Snapshotting {vm} before download...")
                before = snapshot_vm_files(
                    vm=vm,
                    project=args.project,
                    zone=zone,
                    remote_repo=args.remote_repo,
                )
                stream_export_from_vm(
                    vm=vm,
                    project=args.project,
                    zone=zone,
                    remote_repo=args.remote_repo,
                    staging_root=staging_root,
                )
                print(f"Snapshotting {vm} after download...")
                after = snapshot_vm_files(
                    vm=vm,
                    project=args.project,
                    zone=zone,
                    remote_repo=args.remote_repo,
                )
                unstable = discard_unstable_files(
                    staging_root=staging_root,
                    before=before,
                    after=after,
                )
                if unstable:
                    print(
                        f"Skipped {len(unstable)} files on {vm} because they "
                        "changed during download."
                    )
                build_objective_index(staging_root)
                vm_attempts = collect_attempts(staging_root, vm)
                print(f"Found {len(vm_attempts)} attempts on {vm}.")
                result = merge_vm_attempts(
                    vm_attempts,
                    args.destination,
                    vm,
                    unstable,
                )
                print(
                    f"Saved {result['updates_saved_from_last_vm']} new or newer "
                    f"runs from {vm}; {result['trajectory_runs_written']} runs are "
                    "now stored locally."
                )

    print(f"Organized {result['trajectory_runs_written']} runs in " f"{result['root']}")
    print(f"Manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
