from unittest.mock import MagicMock, patch

import pytest

from workflows.detect_workflow import DetectWorkflow
from workflows.exploit_workflow import ExploitWorkflow
from workflows.patch_workflow import PatchWorkflow


@pytest.mark.parametrize(
    ("workflow_class", "phase_target"),
    (
        (DetectWorkflow, "workflows.detect_workflow.DetectPhase"),
        (ExploitWorkflow, "workflows.exploit_workflow.ExploitPhase"),
        (PatchWorkflow, "workflows.patch_workflow.PatchPhase"),
    ),
)
def test_objective_override_reaches_phase(workflow_class, phase_target, tmp_path):
    workflow = workflow_class.__new__(workflow_class)
    workflow.params = {
        "model": "student",
        "phase_iterations": 10,
        "objective_override": "Rewritten objective",
    }
    workflow.task_dir = tmp_path
    workflow.bounty_number = "0"
    workflow.bounty_metadata = {}
    workflow.repo_metadata = {"target_host": "target:8000"}
    workflow.initial_prompt = "{objective_override}\n\nOriginal instructions"
    workflow.interactive = False
    workflow._register_root_phase = MagicMock()

    with (
        patch(phase_target) as phase,
        patch(
            f"{workflow_class.__module__}.read_writeup",
            return_value="Original task",
        ),
    ):
        workflow._create_phases()

    assert phase.call_args.kwargs["objective_override"] == "Rewritten objective"
