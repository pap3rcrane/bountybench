import argparse
import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Type

from rich.console import Console
from rich.traceback import Traceback

from agents.teacher_agent import TeacherAgent, TeacherMode, TeacherSystemPromptPlacement
from utils.logger import get_main_logger, logger_config
from workflows.base_workflow import BaseWorkflow
from workflows.detect_patch_workflow import DetectPatchWorkflow
from workflows.detect_workflow import DetectWorkflow
from workflows.exploit_patch_workflow import ExploitPatchWorkflow
from workflows.exploit_workflow import ExploitWorkflow
from workflows.patch_workflow import PatchWorkflow
from workflows.teacher_modes import (
    rewrite_objective,
    write_objective,
    write_teacher_trace,
)

# This module serves as the single entry point for running workflows via the command line.
# Example usage:
# python -m workflows.runner --workflow-type exploit_patch_workflow \
#     --task_dir bountytasks/lunary \
#     --bounty_number 0 \
#     --model anthropic/claude-3-5-sonnet-20241022 \
#     --phase_iterations 1 \
#     --use_helm


console = Console(record=True)
RUNNER_EVENT_PREFIX = "BOUNTYBENCH_EVENT "


def emit_runner_event(event: str, **fields) -> None:
    """Emit one machine-readable progress event for batch launchers."""
    payload = {
        "event": event,
        "timestamp": datetime.now().astimezone().isoformat(),
        **fields,
    }
    print(RUNNER_EVENT_PREFIX + json.dumps(payload, default=str), flush=True)


import signal


def signal_handler(signum, frame):
    console.print(f"[bold red]Received signal {signum}. Exiting with error.[/]")
    sys.exit(1)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class WorkflowRunner:
    """Centralized workflow runner with error handling and dynamic argument parsing."""

    _workflow_factory: Dict[str, Type[BaseWorkflow]] = {
        "exploit_patch_workflow": ExploitPatchWorkflow,
        "detect_patch_workflow": DetectPatchWorkflow,
        "patch_workflow": PatchWorkflow,
        "detect_workflow": DetectWorkflow,
        "exploit_workflow": ExploitWorkflow,
    }

    def __init__(self):
        self.parser = self._create_parser()
        self.args = None
        self.workflow = None
        self.kwargs = None
        self.error_log_dir = Path("error_logs")

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Run security testing workflows")
        parser.add_argument(
            "--workflow-type",
            type=str,
            required=True,
            choices=list(self._workflow_factory.keys()),
            help="Type of workflow to execute",
        )
        parser.add_argument(
            "--task_dir", type=str, help="Path to the task repository directory"
        )
        parser.add_argument("--bounty_number", type=str, help="Bounty number to target")
        parser.add_argument(
            "--vulnerability_type",
            type=str,
            help="Vulnerability type to detect (for detect_patch_workflow)",
        )
        parser.add_argument(
            "--phase_iterations",
            type=int,
            help="Maximum iterations for workflow phases",
        )
        parser.add_argument(
            "--use_mock_model", action="store_true", help="Use a mock model for testing"
        )
        parser.add_argument("--use_helm", action="store_true", help="Use HelmModels")
        parser.add_argument("--model", type=str, help="LM model to query")
        parser.add_argument(
            "--teacher_model",
            type=str,
            default="openrouter/deepseek/deepseek-v4-pro",
            help="Model used by the TeacherAgent",
        )
        parser.add_argument(
            "--teacher_mode",
            type=str,
            choices=[mode.value for mode in TeacherMode],
            help="Teacher behavior: observe, steer, or objective_rewrite",
        )
        parser.add_argument(
            "--teacher_system_prompt_file",
            type=str,
            help="Path to the TeacherAgent system prompt text file",
        )
        parser.add_argument(
            "--teacher_system_prompt_placement",
            type=str,
            choices=[placement.value for placement in TeacherSystemPromptPlacement],
            help="Required teacher system prompt transport: none, prepend, or system",
        )
        parser.add_argument(
            "--teacher_max_input_tokens",
            type=int,
            default=131072,
            help="Maximum input tokens for the TeacherAgent's full-trace context",
        )
        parser.add_argument(
            "--teacher_max_output_tokens",
            type=int,
            default=4096,
            help="Maximum output tokens for TeacherAgent responses",
        )
        parser.add_argument(
            "--teacher_temperature",
            type=float,
            default=0.0,
            help="Sampling temperature for TeacherAgent responses",
        )
        parser.add_argument(
            "--generate_source_runs",
            action="store_true",
            help="Run three teacher-free students before objective rewriting",
        )
        parser.add_argument(
            "--source_logs",
            nargs=3,
            metavar=("LOG_1", "LOG_2", "LOG_3"),
            help="Exactly three source logs for objective_rewrite",
        )
        parser.add_argument(
            "--objective_output_dir",
            type=str,
            default="generated_objectives",
            help="Directory for generated objective JSON files",
        )
        parser.add_argument(
            "--student_prompt_append_file",
            type=str,
            help="Text file appended to the rendered student benchmark prompt",
        )
        parser.add_argument(
            "--max_input_tokens", type=int, help="Maximum tokens to pass to the model"
        )
        parser.add_argument(
            "--max_output_tokens", type=int, help="Maximum tokens for model output"
        )
        parser.add_argument(
            "--disable_submit",
            action="store_true",
            help="Disable agent from submitting the exploit (not recommended)",
        )

        parser.add_argument(
            "--use_cwe",
            action="store_true",
            help="If set, passes CWE from bounty metadata to the phase",
        )

        parser.add_argument(
            "--logging_level",
            type=str,
            choices=[
                "DEBUG",
                "INFO",
                "STATUS",
                "SUCCESS_STATUS",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ],
            default="DEBUG",
            help="Set the logging level (default: DEBUG)",
        )
        return parser

    def parse_arguments(self) -> None:
        self.args = self.parser.parse_args()
        self._validate_teacher_arguments()

    def _validate_teacher_arguments(self) -> None:
        mode = self.args.teacher_mode
        system_prompt_file = self.args.teacher_system_prompt_file
        system_prompt_placement = self.args.teacher_system_prompt_placement

        if bool(mode) != bool(system_prompt_placement):
            self.parser.error(
                "--teacher_mode and --teacher_system_prompt_placement must be "
                "provided together"
            )
        if system_prompt_placement == TeacherSystemPromptPlacement.NONE.value:
            if system_prompt_file:
                self.parser.error(
                    "--teacher_system_prompt_file must be omitted when "
                    "--teacher_system_prompt_placement is none"
                )
        elif system_prompt_placement and not system_prompt_file:
            self.parser.error(
                "--teacher_system_prompt_file is required when "
                "--teacher_system_prompt_placement is prepend or system"
            )
        elif system_prompt_file and not mode:
            self.parser.error(
                "--teacher_system_prompt_file requires --teacher_mode and "
                "--teacher_system_prompt_placement"
            )
        if (
            system_prompt_placement == TeacherSystemPromptPlacement.SYSTEM.value
            and not self.args.teacher_model.startswith("google/")
        ):
            self.parser.error(
                "--teacher_system_prompt_placement system requires a direct "
                "google/... Gemini teacher model"
            )

        has_source_logs = self.args.source_logs is not None
        if mode == TeacherMode.OBJECTIVE_REWRITE.value:
            if self.args.generate_source_runs == has_source_logs:
                self.parser.error(
                    "objective_rewrite requires exactly one of "
                    "--generate_source_runs or --source_logs LOG_1 LOG_2 LOG_3"
                )
        elif self.args.generate_source_runs or has_source_logs:
            self.parser.error(
                "--generate_source_runs and --source_logs are only valid with "
                "--teacher_mode objective_rewrite"
            )

    def initialize_workflow(self) -> None:
        """Initialize the workflow instance with parsed arguments."""
        # Set logging level early
        level_str = self.args.logging_level.upper()
        level = getattr(logging, level_str, logging.INFO)
        logger_config.set_global_log_level(level)

        workflow_class = self._workflow_factory[self.args.workflow_type]

        # Convert parsed args to kwargs
        kwargs = vars(self.args).copy()
        # Remove workflow-type as it's not needed for workflow instantiation
        kwargs.pop("workflow_type")

        # Remove None values from kwargs
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Handle path conversions
        for arg in ["task_dir", "log_dir"]:
            if arg in kwargs and kwargs[arg] is not None:
                kwargs[arg] = Path(kwargs[arg])

        self.kwargs = kwargs
        if self.args.teacher_mode == TeacherMode.OBJECTIVE_REWRITE.value:
            workflow_validator = object.__new__(workflow_class)
            workflow_validator.validate_arguments(self._student_kwargs(kwargs))
            if self.args.teacher_system_prompt_file:
                TeacherAgent._resolve_prompt_path(self.args.teacher_system_prompt_file)
        else:
            self.workflow = workflow_class(**kwargs)

    @staticmethod
    def _student_kwargs(kwargs: dict) -> dict:
        student_kwargs = kwargs.copy()
        for key in (
            "teacher_mode",
            "teacher_system_prompt_file",
            "teacher_system_prompt_placement",
            "teacher_model",
            "teacher_max_input_tokens",
            "teacher_max_output_tokens",
            "teacher_temperature",
            "generate_source_runs",
            "source_logs",
            "objective_output_dir",
        ):
            student_kwargs.pop(key, None)
        return student_kwargs

    async def _run_initialized_workflow(self, run_role: str = "normal") -> Path:
        emit_runner_event("student_run_started", run_role=run_role)
        try:
            await self.workflow.run()
            if not self.check_workflow_completion():
                raise Exception("Workflow marked as incomplete in the log file.")
        except Exception as error:
            log_file = getattr(
                getattr(self.workflow, "workflow_message", None), "log_file", None
            )
            emit_runner_event(
                "student_run_finished",
                run_role=run_role,
                status="failure",
                workflow_log_path=log_file,
                error=f"{error.__class__.__name__}: {error}",
            )
            raise

        log_file = Path(self.workflow.workflow_message.log_file)
        emit_runner_event(
            "student_run_finished",
            run_role=run_role,
            status="success",
            workflow_log_path=log_file,
            error=None,
        )
        return log_file

    async def _run_objective_rewrite(self) -> None:
        workflow_class = self._workflow_factory[self.args.workflow_type]
        student_kwargs = self._student_kwargs(self.kwargs)
        system_prompt_file = (
            TeacherAgent._resolve_prompt_path(self.args.teacher_system_prompt_file)
            if self.args.teacher_system_prompt_file
            else None
        )

        if self.args.generate_source_runs:
            source_logs = []
            for run_number in range(1, 4):
                console.print(
                    f"[bold cyan]Running teacher-free source student "
                    f"{run_number}/3...[/]"
                )
                self.workflow = workflow_class(**student_kwargs)
                source_logs.append(
                    await self._run_initialized_workflow(
                        run_role=f"source_{run_number}"
                    )
                )
        else:
            source_logs = [Path(path) for path in self.args.source_logs]
            missing_logs = [str(path) for path in source_logs if not path.is_file()]
            if missing_logs:
                raise ValueError(
                    "Source log files do not exist: " + ", ".join(missing_logs)
                )
            if len({path.resolve() for path in source_logs}) != 3:
                raise ValueError("--source_logs must name three different log files")

        emit_runner_event("teacher_rewrite_started")
        try:
            rewrite_result = await rewrite_objective(
                source_logs=source_logs,
                system_prompt_file=system_prompt_file,
                system_prompt_placement=TeacherSystemPromptPlacement(
                    self.args.teacher_system_prompt_placement
                ),
                model=self.args.teacher_model,
                use_mock_model=self.args.use_mock_model,
                max_input_tokens=self.args.teacher_max_input_tokens,
                max_output_tokens=self.args.teacher_max_output_tokens,
                temperature=self.args.teacher_temperature,
            )
        except Exception as error:
            emit_runner_event(
                "teacher_rewrite_finished",
                status="failure",
                error=f"{error.__class__.__name__}: {error}",
            )
            raise
        objective_file = write_objective(
            objective=rewrite_result.objective,
            output_dir=Path(self.args.objective_output_dir),
            task_dir=Path(self.args.task_dir),
            bounty_number=self.args.bounty_number,
            workflow_type=self.args.workflow_type,
        )
        teacher_trace_file = write_teacher_trace(
            teacher_trace=rewrite_result.teacher_trace,
            objective_path=objective_file,
        )
        emit_runner_event(
            "teacher_rewrite_finished",
            status="success",
            error=None,
            objective_path=objective_file,
            teacher_trace_path=teacher_trace_file,
        )
        console.print(f"[bold green]Wrote rewritten objective: {objective_file}[/]")
        console.print(f"[bold green]Wrote teacher trace: {teacher_trace_file}[/]")

    async def run(self) -> int:
        """Execute the workflow with error handling."""
        try:
            console.print(f"[bold green]*" * 40)
            console.print(
                f"[bold green]Starting {self.args.workflow_type} workflow...[/]"
            )
            console.print(f"[bold green]*" * 40)
            if self.args.teacher_mode == TeacherMode.OBJECTIVE_REWRITE.value:
                await self._run_objective_rewrite()
            else:
                await self._run_initialized_workflow(run_role="normal")

            console.print(f"[bold green]*" * 40)
            console.print(
                f"[bold green]Completed {self.args.workflow_type} workflow[/]"
            )
            console.print(f"[bold green]*" * 40)
            return 0
        except Exception as e:
            console.print(f"[bold red]*" * 40)
            console.print(
                f"[bold red]Error in {self.args.workflow_type} workflow:[/] {e}"
            )
            console.print(f"[bold red]*" * 40)
            console.print(Traceback())

            # Create error report
            self.create_error_report(e)

            sys.exit(1)

    def check_workflow_completion(self) -> bool:
        """Check if the workflow is marked as complete in the log file."""
        try:
            log_file_path = Path(self.workflow.workflow_message.log_file)
            if not log_file_path.exists():
                console.print(
                    f"[bold yellow]Warning: Log file not found: {log_file_path}[/]"
                )
                return False

            with log_file_path.open("r") as log_file:
                log_data = json.load(log_file)
                workflow_complete = (
                    log_data.get("workflow_metadata", {})
                    .get("workflow_summary", {})
                    .get("complete", False)
                )
                if not workflow_complete:
                    console.print(
                        "[bold red]Workflow marked as incomplete in the log file.[/]"
                    )
                    return False
                return True
        except Exception as e:
            console.print(f"[bold red]Error checking workflow completion: {e}[/]")
            return False

    def create_error_report(self, error: Exception) -> None:
        """Create an error report file with the error details and console output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create error log directory if it doesn't exist
        self.error_log_dir.mkdir(exist_ok=True, parents=True)

        error_file = self.error_log_dir / f"error_report_{timestamp}.txt"

        log_file = None
        if (
            self.workflow
            and hasattr(self.workflow, "workflow_message")
            and hasattr(self.workflow.workflow_message, "log_file")
        ):
            log_file = Path(self.workflow.workflow_message.log_file)

        with error_file.open("w") as f:
            f.write("ERROR REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Workflow Type: {self.args.workflow_type}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if log_file:
                f.write(f"Log File: {log_file}\n")
            f.write("\n")

            f.write("Workflow Arguments:\n")
            f.write("-" * 50 + "\n")
            for key, value in self.kwargs.items():
                f.write(f"{key}: {str(value)}\n")
            f.write("\n")

            f.write("Error Details:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{error.__class__.__name__}: {str(error)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "=" * 50 + "\n\n")

            # Include recent logs from the buffer
            f.write("Recent Logs:\n")
            f.write("-" * 50 + "\n")
            for log_entry in logger_config.log_buffer_handler.get_logs():
                f.write(log_entry + "\n")
            f.write("=" * 50 + "\n\n")  # End of highlighted section

            if log_file:
                f.write("Workflow Log File:\n")
                f.write("-" * 50 + "\n")
                log_file_path = Path(self.workflow.workflow_message.log_file)
                if log_file_path.exists():
                    with log_file_path.open("r") as log_file:
                        f.write(log_file.read())
                else:
                    f.write(f"Log file not found: {log_file_path}\n")
                f.write("\n" + "=" * 50 + "\n\n")

            f.write("Console Output:\n")
            f.write("-" * 50 + "\n")
            f.write(console.export_text())

        console.print(f"[bold yellow]Error report created: {error_file}[/]")


async def main() -> int:
    """Main entry point for workflow execution."""
    runner = WorkflowRunner()
    try:
        runner.parse_arguments()
        runner.initialize_workflow()
        return await runner.run()
    except Exception as e:
        console.print(f"[bold red]Initialization error:[/] {e}")
        console.print(Traceback())
        runner.create_error_report(e)
        sys.exit(1)
    except SystemExit:
        raise


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
