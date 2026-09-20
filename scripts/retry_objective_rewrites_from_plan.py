#!/usr/bin/env python3
"""Retry objective rewrites from saved source logs without rerunning students."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f%z")


def find_repository_root(script_path: Path) -> Path:
    """Find the checkout even when this helper is copied into an input bundle."""
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "workflows" / "runner.py").is_file():
            return candidate
    raise ValueError(
        f"Could not locate the BountyBench repository above {script_path}"
    )


def load_plan(path: Path) -> list[dict]:
    records = []
    with path.open() as plan_file:
        for line_number, line in enumerate(plan_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            source_logs = record.get("source_logs")
            if not isinstance(source_logs, list) or len(source_logs) != 3:
                raise ValueError(
                    f"Plan record at {path}:{line_number} requires three source_logs"
                )
            records.append(record)
    return records


def configuration_name(record: dict) -> str:
    return "_".join(
        [
            record["repo_name"],
            f"bounty_{record['bounty_number']}",
            record["workflow_type"],
            record["system_prompt_name"],
            record["system_prompt_placement"],
            f"repeat_{record['run_number']}",
        ]
    )


def validate_sources(records: list[dict]) -> None:
    missing = []
    for record in records:
        for source_log in record["source_logs"]:
            if not Path(source_log).is_file():
                missing.append(f"{configuration_name(record)}: {source_log}")
    if missing:
        sample = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n...and {len(missing) - 20} more"
        raise ValueError(f"Missing source logs:\n{sample}{suffix}")


def build_command(
    record: dict,
    *,
    objective_directory: Path,
    teacher_model: str,
    teacher_max_input_tokens: int,
    teacher_max_output_tokens: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "workflows.runner",
        "--workflow-type",
        record["workflow_type"],
        "--task_dir",
        f"bountytasks/{record['repo_name']}",
        "--bounty_number",
        str(record["bounty_number"]),
        "--model",
        "openrouter/deepseek/deepseek-chat-v3-0324",
        "--teacher_model",
        teacher_model,
        "--teacher_mode",
        "objective_rewrite",
        "--teacher_system_prompt_placement",
        record["system_prompt_placement"],
        "--teacher_max_input_tokens",
        str(teacher_max_input_tokens),
        "--teacher_max_output_tokens",
        str(teacher_max_output_tokens),
        "--source_logs",
        *record["source_logs"],
        "--objective_output_dir",
        str(objective_directory),
        "--phase_iterations",
        "100",
        "--max_input_tokens",
        "8192",
        "--max_output_tokens",
        "8192",
        "--logging_level",
        "DEBUG",
    ]
    prompt_file = record.get("system_prompt_file")
    if prompt_file:
        command.extend(["--teacher_system_prompt_file", prompt_file])
    return command


def run_one(
    record: dict,
    *,
    repository_root: Path,
    log_directory: Path,
    objective_root: Path,
    teacher_model: str,
    teacher_max_input_tokens: int,
    teacher_max_output_tokens: int,
    dry_run: bool,
) -> dict:
    name = configuration_name(record)
    log_path = log_directory / f"{name}.log"
    objective_directory = objective_root / name
    objective_directory.mkdir(parents=True, exist_ok=True)
    command = build_command(
        record,
        objective_directory=objective_directory,
        teacher_model=teacher_model,
        teacher_max_input_tokens=teacher_max_input_tokens,
        teacher_max_output_tokens=teacher_max_output_tokens,
    )
    started_at = datetime.now().astimezone()
    if dry_run:
        return_code = 0
        error = None
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n")
    else:
        with log_path.open("w") as log_file:
            result = subprocess.run(
                command,
                cwd=repository_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = result.returncode
        error = None if return_code == 0 else f"Teacher runner exited {return_code}"
    finished_at = datetime.now().astimezone()
    return {
        "record_type": "configuration",
        "run_role": "configuration",
        "teacher_only_retry": True,
        "source_runs_reused": True,
        "status": "success" if return_code == 0 else "failure",
        "exit_code": return_code,
        "error": error,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "log_path": str(log_path),
        "objective_directory": str(objective_directory),
        **record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--teacher-model", default="google/gemini-3.6-flash")
    parser.add_argument("--teacher-max-input-tokens", type=int, default=1048576)
    parser.add_argument("--teacher-max-output-tokens", type=int, default=65536)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be positive")

    repository_root = find_repository_root(Path(__file__).resolve())
    plan_path = args.plan.expanduser().resolve()
    records = load_plan(plan_path)
    validate_sources(records)
    if args.validate_only:
        print(f"Validated {len(records)} teacher-only retries from {plan_path}")
        print("All three saved source logs exist for every configuration.")
        return 0

    run_id = f"objective_rewrite_teacher_only_{timestamp()}"
    log_directory = repository_root / "batch_logs" / "objective_rewrite_teacher_only" / run_id
    objective_root = repository_root / "generated_objectives" / "teacher_only_retries" / run_id
    status_directory = repository_root / "runs" / "objective_rewrite_teacher_only" / run_id
    status_file = status_directory / "run_status.jsonl"
    log_directory.mkdir(parents=True, exist_ok=True)
    objective_root.mkdir(parents=True, exist_ok=True)
    status_directory.mkdir(parents=True, exist_ok=True)

    print(f"Plan: {plan_path}")
    print(f"Configurations: {len(records)}")
    print(f"Concurrency: {args.jobs}")
    print(f"Status: {status_file}")
    print("Student source runs: reused; no students will be launched")

    successes = 0
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_one,
                record,
                repository_root=repository_root,
                log_directory=log_directory,
                objective_root=objective_root,
                teacher_model=args.teacher_model,
                teacher_max_input_tokens=args.teacher_max_input_tokens,
                teacher_max_output_tokens=args.teacher_max_output_tokens,
                dry_run=args.dry_run,
            ): record
            for record in records
        }
        with status_file.open("a") as status_output:
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                record = future.result()
                status_output.write(json.dumps(record, sort_keys=True) + "\n")
                status_output.flush()
                if record["status"] == "success":
                    successes += 1
                else:
                    failures += 1
                print(
                    f"[{completed}/{len(records)}] {record['status'].upper()}: "
                    f"{configuration_name(record)}"
                )

    print(f"Finished: success={successes} failure={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
