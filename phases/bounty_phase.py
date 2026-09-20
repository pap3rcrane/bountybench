import os
import subprocess
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from agents.base_agent import AgentConfig, BaseAgent
from agents.teacher_agent import (
    TeacherAgent,
    TeacherAgentConfig,
    TeacherMode,
    TeacherSystemPromptPlacement,
)
from phases.base_phase import BasePhase
from prompts.prompts import SUBMISSION_INSTRUCTIONS
from prompts.vulnerability_prompts import get_specialized_instructions
from resources.base_resource import BaseResourceConfig
from resources.model_resource.model_resource import ModelResourceConfig
from resources.resource_type import ResourceType
from utils.logger import get_main_logger
from workflows.base_workflow import BaseWorkflow
from workflows.workflow_context import current_workflow_id

logger = get_main_logger(__name__)


class BountyPhase(BasePhase, ABC):
    AGENT_CLASSES: List[Type[BaseAgent]] = []

    def __init__(self, workflow: "BaseWorkflow", **kwargs):
        """
        Args:
            workflow (BaseWorkflow): The parent workflow.
            **kwargs: Additional keyword arguments.
        """
        self.workflow_id = current_workflow_id.get()

        self.model: str = kwargs.get("model", "")
        self.helm: Any = kwargs.get("helm")
        self.bounty_number: str = kwargs.get("bounty_number", "")
        self.use_mock_model: str = kwargs.get("use_mock_model")

        self._initial_prompt: str = kwargs.get("initial_prompt", "")
        self.bounty_dir: Path = (
            workflow.task_dir / "bounties" / f"bounty_{self.bounty_number}"
        )

        self.tmp_dir: Path = (
            workflow.task_dir
            / "bounties"
            / f"bounty_{self.bounty_number}"
            / f"tmp_{workflow.workflow_message.workflow_id}"
        )

        self.output_agent_files_name: str = f"agent_created_files_{self.workflow_id}"

        self.submit = kwargs.get("submit", True)

        super().__init__(workflow, **kwargs)

    def teacher_agent_config(self) -> TeacherAgentConfig:
        return TeacherAgentConfig(
            system_prompt_file=self.workflow.params.get("teacher_system_prompt_file"),
            system_prompt_placement=TeacherSystemPromptPlacement(
                self.workflow.params["teacher_system_prompt_placement"]
            ),
            mode=TeacherMode(self.workflow.params["teacher_mode"]),
        )

    @property
    def has_scheduled_teacher(self) -> bool:
        return self.workflow.params.get("teacher_mode") in {
            TeacherMode.OBSERVE.value,
            TeacherMode.STEER.value,
        }

    def add_teacher_agent(
        self,
        agents: Dict[str, Tuple[Type[BaseAgent], Optional[AgentConfig]]],
    ) -> None:
        if self.has_scheduled_teacher:
            agents["teacher_agent"] = (TeacherAgent, self.teacher_agent_config())

    def add_teacher_resource(
        self,
        resource_configs: List[Tuple[ResourceType, BaseResourceConfig]],
    ) -> None:
        if self.has_scheduled_teacher:
            resource_configs.append(
                (ResourceType.TEACHER_MODEL, self.teacher_model_config())
            )

    def teacher_model_config(self) -> ModelResourceConfig:
        model = self.workflow.params.get(
            "teacher_model", "openrouter/deepseek/deepseek-v4-pro"
        )
        return ModelResourceConfig.create(
            model=model,
            use_helm=False,
            use_mock_model=self.workflow.params.get("use_mock_model", False),
            max_input_tokens=self.workflow.params.get(
                "teacher_max_input_tokens", 131072
            ),
            max_output_tokens=self.workflow.params.get(
                "teacher_max_output_tokens", 4096
            ),
            temperature=self.workflow.params.get("teacher_temperature", 0.0),
            preserve_newest_input=True,
            thinking_level="high" if model.startswith("google/") else None,
        )

    def _create_initial_agent_message(self) -> None:
        """Create the initial agent message for the bounty phase."""
        if self.params.get("task_dir"):
            # Start with the highest depth and decrease if needed
            # Filter out common nonsensical directories
            for depth in [4, 3, 2]:
                codebase_structure = subprocess.run(
                    [
                        "tree",
                        "-L",
                        str(depth),
                        "-I",
                        "_*|venv|node_modules|.git|__pycache__|.idea|.vscode",
                    ],
                    cwd=str(self.tmp_dir),
                    capture_output=True,
                    text=True,
                ).stdout

                tokens = len(codebase_structure.split())

                # If we're under 300 tokens, use this depth
                if tokens < 300:
                    break

                # If we've reached minimum depth and still too large, fallback to ls
                if depth == 2:
                    # Try using ls -la as a fallback
                    codebase_structure = subprocess.run(
                        ["ls", "-la"],
                        cwd=str(self.tmp_dir),
                        capture_output=True,
                        text=True,
                    ).stdout

            # Add the command that was used to the output
            command = (
                f"tree -L {depth} -I '_*|venv|node_modules|.git|__pycache__|.idea|.vscode'"
                if tokens < 300
                else "ls -la"
            )
            self.params["codebase"] = f"$ {command}\n{codebase_structure}"

        self.params["specialized_instructions"] = (
            get_specialized_instructions(self.params.get("vulnerability_type"))
            if self.params.get("vulnerability_type")
            else ""
        )

        if self.params.get("vulnerability_type"):
            logger.error("vulnerability_type found - unsupported feature")
            self.params["vulnerability_type"] += " "

        if self.params.get("submit"):
            self.params["submit"] = SUBMISSION_INSTRUCTIONS

        super()._create_initial_agent_message()
