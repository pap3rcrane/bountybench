#!/usr/bin/env python3
"""Export the latest workflow log as a compact agent/environment timeline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
SECTION_RE = re.compile(r"(?m)^(Reflection|Plan and Status|Thought|Log|Command):\s*")


def newest_file(directory: Path, pattern: str) -> Path | None:
    files = [path for path in directory.rglob(pattern) if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None


def unique_output_path(path: Path) -> Path:
    """Return a non-existing path without replacing a previous export."""
    if not path.exists():
        return path
    while True:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f%z")
        candidate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        if not candidate.exists():
            return candidate


def automatic_output_path(data: dict[str, Any], generated_at: datetime) -> Path:
    workflow = data.get("workflow_metadata", {})
    task = workflow.get("task", {})
    task_name = Path(str(task.get("task_dir", "unknown_task"))).name
    bounty_number = str(task.get("bounty_number", "unknown"))
    workflow_name = str(workflow.get("workflow_name", "unknown_workflow"))
    timestamp = generated_at.strftime("%Y-%m-%d_%H-%M-%S-%f%z")
    filename = (
        f"{task_name}_bounty_{bounty_number}_{workflow_name}_{timestamp}_"
        "environment_timeline.json"
    )
    return OUTPUT_DIR / filename


def structured_log_for(full_log: Path) -> Path | None:
    """Find the structured JSON log corresponding to a full text log."""
    try:
        relative = full_log.resolve().relative_to(
            (PROJECT_ROOT / "full_logs").resolve()
        )
    except ValueError:
        relative = None

    if relative is not None:
        candidate = PROJECT_ROOT / "logs" / relative.with_suffix(".json")
        if candidate.is_file():
            return candidate

    matches = list((PROJECT_ROOT / "logs").rglob(f"{full_log.stem}.json"))
    return matches[0] if matches else None


def selectable_logs() -> list[Path]:
    full_logs = sorted(
        (
            path
            for path in (PROJECT_ROOT / "full_logs").rglob("*.log")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    usable_full_logs = [path for path in full_logs if structured_log_for(path)]
    if usable_full_logs:
        return usable_full_logs
    return sorted(
        (path for path in (PROJECT_ROOT / "logs").rglob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def select_log_interactively() -> Path:
    logs = selectable_logs()
    if not logs:
        raise FileNotFoundError("No workflow logs were found under logs/ or full_logs/")

    print("Available workflow logs (newest first):")
    for index, path in enumerate(logs, start=1):
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        relative = path.relative_to(PROJECT_ROOT)
        print(f"{index:>3}. {modified:%Y-%m-%d %H:%M:%S %z}  {relative}")

    while True:
        choice = input(f"Select a log [1-{len(logs)}] or q to cancel: ").strip()
        if choice.lower() in {"q", "quit"}:
            raise SystemExit("Selection cancelled.")
        try:
            selected_index = int(choice) - 1
        except ValueError:
            selected_index = -1
        if 0 <= selected_index < len(logs):
            return logs[selected_index]
        print("Invalid selection. Enter one of the listed numbers.")


def resolve_input(path_arg: str | None) -> tuple[Path, Path | None]:
    """Return (structured JSON log, corresponding full log if known)."""
    if path_arg:
        selected = Path(path_arg).expanduser()
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
        selected = selected.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Log does not exist: {selected}")
        if selected.suffix == ".json":
            return selected, None
        if selected.suffix == ".log":
            structured = structured_log_for(selected)
            if structured is None:
                raise FileNotFoundError(
                    f"No structured JSON log corresponds to {selected}"
                )
            return structured, selected
        raise ValueError("Input must be a workflow .json or full .log file")

    latest_full = newest_file(PROJECT_ROOT / "full_logs", "*.log")
    if latest_full is not None:
        structured = structured_log_for(latest_full)
        if structured is not None:
            return structured, latest_full

    latest_json = newest_file(PROJECT_ROOT / "logs", "*.json")
    if latest_json is None:
        raise FileNotFoundError("No workflow logs were found under logs/ or full_logs/")
    return latest_json, None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def compact_system_message(message: str) -> str:
    """Keep task/environment context but omit response-format boilerplate."""
    marker = "\n\nAlways respond in this format exactly:"
    return clean_text(message.split(marker, 1)[0])


def parse_agent_response(message: str) -> tuple[str, str | None]:
    """Remove the cumulative command log and return (decision, command)."""
    text = clean_text(message)
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return text.removesuffix("<END>").rstrip(), None

    kept_sections: list[str] = []
    command: str | None = None
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = match.group(1)
        value = text[match.end() : end].strip()
        if name == "Log":
            continue
        if name == "Command":
            command = value.removesuffix("<END>").rstrip() or None
            continue
        kept_sections.append(f"{name}: {value}")
    return "\n\n".join(kept_sections), command


def shortened(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit].rstrip()}\n...[{omitted} characters omitted]"


def add_step(
    steps: list[dict[str, Any]],
    *,
    source: str,
    message_type: str,
    message: str,
    iteration: int | None,
    timestamp: str | None,
    command: str | None = None,
    exit_code: int | None = None,
) -> None:
    step: dict[str, Any] = {
        "step_number": len(steps) + 1,
        "source": source,
        "message_type": message_type,
        "iteration": iteration,
        "timestamp": timestamp,
        "message": message,
    }
    if command:
        step["command"] = command
    if exit_code is not None:
        step["exit_code"] = exit_code
    steps.append(step)


def action_message(
    steps: list[dict[str, Any]],
    agent: dict[str, Any],
    action: dict[str, Any],
    max_environment_chars: int,
) -> None:
    agent_id = agent.get("agent_id", "unknown_agent")
    resource_id = action.get("resource_id", "unknown_resource")
    source = f"{agent_id}/{resource_id}"
    metadata = action.get("additional_metadata") or {}
    message = clean_text(action.get("message"))
    command = clean_text(action.get("command") or metadata.get("command")) or None
    exit_code = metadata.get("exit_code")

    if resource_id == "model":
        message, parsed_command = parse_agent_response(message)
        command = parsed_command or command
        kind = "agent_decision"
    elif resource_id == "submission":
        message = message or "Submitted the agent's work for evaluation."
        kind = "submission"
    else:
        message = shortened(message, max_environment_chars)
        if not message:
            message = "Command completed with no output."
        kind = "environment_result"

    add_step(
        steps,
        source=source,
        message_type=kind,
        iteration=agent.get("iteration"),
        timestamp=action.get("timestamp") or agent.get("timestamp"),
        message=message,
        command=command,
        exit_code=exit_code,
    )


def build_timeline(
    data: dict[str, Any], max_environment_chars: int
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for phase in data.get("phase_messages", []):
        for agent in phase.get("agent_messages", []):
            agent_id = agent.get("agent_id", "unknown_agent")
            message = clean_text(agent.get("message"))

            if agent_id == "system":
                if message:
                    add_step(
                        steps,
                        source="system",
                        message_type="task_instruction",
                        iteration=agent.get("iteration"),
                        timestamp=agent.get("timestamp"),
                        message=compact_system_message(message),
                    )
                continue

            actions = agent.get("action_messages") or []
            for action in actions:
                action_message(steps, agent, action, max_environment_chars)

            if message:
                add_step(
                    steps,
                    source=agent_id,
                    message_type="agent_status",
                    iteration=agent.get("iteration"),
                    timestamp=agent.get("timestamp"),
                    message=message,
                )
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        help="Workflow .json or full .log file (default: newest full log)",
    )
    input_group.add_argument(
        "--select",
        action="store_true",
        help="Interactively select a workflow log from a newest-first list",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSON path (default: environment_timelines/<task>_bounty_"
            "<number>_<workflow-name>_<exact-date-time>_environment_timeline.json)"
        ),
    )
    parser.add_argument(
        "--max-environment-chars",
        type=int,
        default=6000,
        help="Maximum characters per environment result; 0 keeps everything",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_input = str(select_log_interactively()) if args.select else args.input
    structured_log, full_log = resolve_input(selected_input)
    with structured_log.open(encoding="utf-8") as file:
        data = json.load(file)

    generated_at = datetime.now().astimezone()
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = OUTPUT_DIR / output
    else:
        output = automatic_output_path(data, generated_at)
    output = unique_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    timeline = {
        "source_log": str((full_log or structured_log).resolve()),
        "structured_source_log": str(structured_log.resolve()),
        "workflow": data.get("workflow_metadata", {}),
        "generated_at": generated_at.isoformat(),
        "step_count": 0,
        "steps": build_timeline(data, args.max_environment_chars),
    }
    timeline["step_count"] = len(timeline["steps"])

    with output.open("w", encoding="utf-8") as file:
        json.dump(timeline, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Exported {timeline['step_count']} steps from {structured_log}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
