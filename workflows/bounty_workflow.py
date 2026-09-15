from abc import ABC

from agents.teacher_agent import TeacherMode, TeacherSystemPromptPlacement
from resources.utils import read_bounty_metadata, read_repo_metadata
from utils.logger import get_main_logger
from workflows.base_workflow import BaseWorkflow
from workflows.utils import setup_shared_network

logger = get_main_logger(__name__)


class BountyWorkflow(BaseWorkflow, ABC):
    def validate_arguments(self, kwargs):
        """
        Base validation for all bounty workflows
        """
        super().validate_arguments(kwargs)

        # All bounty workflows require task_dir and bounty_number
        required_base_args = ["task_dir", "bounty_number"]
        missing_args = [arg for arg in required_base_args if arg not in kwargs]
        if missing_args:
            raise ValueError(
                f"Missing required arguments for {self.name}: {', '.join(missing_args)}"
            )

        teacher_mode = kwargs.get("teacher_mode")
        teacher_system_prompt_file = kwargs.get("teacher_system_prompt_file")
        teacher_system_prompt_placement = kwargs.get("teacher_system_prompt_placement")
        if bool(teacher_mode) != bool(teacher_system_prompt_placement):
            raise ValueError(
                "--teacher_mode and --teacher_system_prompt_placement must be "
                "provided together"
            )
        if teacher_mode and teacher_mode not in {mode.value for mode in TeacherMode}:
            raise ValueError(f"Unknown teacher mode: {teacher_mode}")
        if teacher_system_prompt_placement and teacher_system_prompt_placement not in {
            placement.value for placement in TeacherSystemPromptPlacement
        }:
            raise ValueError(
                "Unknown teacher system prompt placement: "
                f"{teacher_system_prompt_placement}"
            )
        if teacher_system_prompt_placement == TeacherSystemPromptPlacement.NONE.value:
            if teacher_system_prompt_file:
                raise ValueError(
                    "--teacher_system_prompt_file must be omitted when "
                    "--teacher_system_prompt_placement is none"
                )
        elif teacher_system_prompt_placement and not teacher_system_prompt_file:
            raise ValueError(
                "--teacher_system_prompt_file is required when "
                "--teacher_system_prompt_placement is prepend or system"
            )
        elif teacher_system_prompt_file and not teacher_mode:
            raise ValueError(
                "--teacher_system_prompt_file requires --teacher_mode and "
                "--teacher_system_prompt_placement"
            )
        if (
            teacher_system_prompt_placement == TeacherSystemPromptPlacement.SYSTEM.value
            and not kwargs.get(
                "teacher_model", "openrouter/deepseek/deepseek-v4-pro"
            ).startswith("google/")
        ):
            raise ValueError(
                "--teacher_system_prompt_placement system requires a direct "
                "google/... Gemini teacher model"
            )

    def _initialize(self):
        self.task_dir = self.params["task_dir"]
        self.bounty_number = self.params["bounty_number"]
        logger.info(
            f"Setting bounty parameters:\nTask dir: {self.task_dir}\nBounty number: {self.bounty_number}"
        )
        self.repo_metadata = read_repo_metadata(self.task_dir)
        self.bounty_metadata = read_bounty_metadata(self.task_dir, self.bounty_number)

        setup_shared_network()

    def _get_metadata(self):
        metadata = {
            "repo_metadata": self.repo_metadata,
            "bounty_metadata": self.bounty_metadata,
        }
        if self.params.get("objective_file"):
            metadata["objective_file"] = self.params["objective_file"]
        if self.params.get("objective_rewrite"):
            metadata["objective_rewrite"] = self.params["objective_rewrite"]
        return metadata

    def _get_task(self):
        return {"task_dir": self.task_dir, "bounty_number": self.bounty_number}
