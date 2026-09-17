#!/usr/bin/env python3
"""Create an exclusion manifest that retries only unresolved matrix failures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUNNER_EVENT_PREFIX = "BOUNTYBENCH_EVENT "
TEACHER_RESPONSE_HEADER = "TEACHER RESPONSE (objective_rewrite)"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONFIGURATION_FIELDS = (
    "teacher_type",
    "repo_name",
    "bounty_number",
    "workflow_type",
    "system_prompt_name",
    "system_prompt_placement",
    "run_number",
)


def _nonempty_teacher_response(lines: list[str]) -> bool:
    response = "\n".join(lines).replace("<END>", "").strip()
    if response.startswith("```") and response.endswith("```"):
        response_lines = response.splitlines()
        if len(response_lines) >= 2:
            response = "\n".join(response_lines[1:-1]).strip()
    return bool(response)


def _teacher_produced_rewrite(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    response_lines = []
    collecting_response = False
    with log_path.open(errors="replace") as log_file:
        for raw_line in log_file:
            line = ANSI_ESCAPE.sub("", raw_line)
            if TEACHER_RESPONSE_HEADER in line:
                collecting_response = True
                response_lines = []
                continue
            if collecting_response:
                stripped = line.strip()
                if stripped and set(stripped) == {"="}:
                    if _nonempty_teacher_response(response_lines):
                        return True
                    collecting_response = False
                    response_lines = []
                else:
                    response_lines.append(line.rstrip())

            marker = line.find(RUNNER_EVENT_PREFIX)
            if marker < 0:
                continue
            payload = line[marker + len(RUNNER_EVENT_PREFIX) :].strip()
            try:
                event, _ = json.JSONDecoder().raw_decode(payload)
            except json.JSONDecodeError:
                continue
            if (
                event.get("event") == "teacher_rewrite_finished"
                and event.get("status") == "success"
            ):
                return True
    return collecting_response and _nonempty_teacher_response(response_lines)


def build_exclusions(status_path: Path) -> tuple[list[dict], dict[str, int]]:
    exclusions = []
    counts = {
        "successful": 0,
        "failed_to_retry": 0,
        "already_skipped": 0,
        "completed_objective_rewrites": 0,
    }
    with status_path.open() as status_file:
        for line_number, line in enumerate(status_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {status_path}:{line_number}") from error
            if record.get("record_type") != "configuration":
                continue
            missing = [field for field in CONFIGURATION_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"Configuration at {status_path}:{line_number} is missing "
                    + ", ".join(missing)
                )

            status = record.get("status")
            if status == "success":
                counts["successful"] += 1
                continue
            if status == "skipped":
                counts["already_skipped"] += 1
                exclusions.append(record)
                continue

            is_completed_rewrite = False
            if record.get("teacher_type") == "objective_rewrite":
                raw_log_path = record.get("configuration_log_path")
                if raw_log_path:
                    log_path = Path(raw_log_path).expanduser()
                    if not log_path.is_absolute():
                        log_path = status_path.parents[3] / log_path
                    is_completed_rewrite = _teacher_produced_rewrite(log_path)

            if is_completed_rewrite:
                counts["completed_objective_rewrites"] += 1
                exclusions.append(record)
            else:
                counts["failed_to_retry"] += 1

    return exclusions, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exclude prior skipped configurations and objective-rewrite failures "
            "whose teacher rewrite already completed. Use the original status file "
            "with --skip-configurations-from to skip successes."
        )
    )
    parser.add_argument("status_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    status_path = args.status_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()
    exclusions, counts = build_exclusions(status_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        for record in exclusions:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Status source: {status_path}")
    print(f"Exclusion manifest: {output_path}")
    print(f"Prior successes (skip via source status): {counts['successful']}")
    print(f"Prior skipped configurations excluded: {counts['already_skipped']}")
    print(
        "Completed teacher rewrites excluded: "
        f"{counts['completed_objective_rewrites']}"
    )
    print(f"Failed configurations to retry: {counts['failed_to_retry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
