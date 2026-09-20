from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from messages.phase_messages.phase_message import PhaseMessage
from messages.workflow_message import WorkflowMessage

# Constants
QUERY_TIME_TAKEN_IN_MS = "query_time_taken_in_ms"
INPUT_TOKEN = "input_token"
OUTPUT_TOKEN = "output_token"


def test_get_total_usage():
    """
    Test the get_total_usage method to ensure it correctly aggregates token usage.
    """
    phase_message_1 = MagicMock(spec=PhaseMessage)
    phase_message_1.usage = {
        INPUT_TOKEN: 300,
        OUTPUT_TOKEN: 150,
        QUERY_TIME_TAKEN_IN_MS: 500,
    }

    phase_message_2 = MagicMock(spec=PhaseMessage)
    phase_message_2.usage = {
        INPUT_TOKEN: 200,
        OUTPUT_TOKEN: 100,
        QUERY_TIME_TAKEN_IN_MS: 700,
    }

    workflow_message = WorkflowMessage("test_workflow")
    workflow_message._phase_messages = [phase_message_1, phase_message_2]

    usage = workflow_message.get_total_usage()

    assert usage == {
        "total_input_tokens": 500,
        "total_output_tokens": 250,
        "total_query_time_taken_in_ms": 1200,
    }
    assert workflow_message.usage == usage


def test_get_total_usage_ignores_missing_and_malformed_phase_values():
    valid_phase = MagicMock(spec=PhaseMessage)
    valid_phase.usage = {
        INPUT_TOKEN: 10,
        OUTPUT_TOKEN: None,
        QUERY_TIME_TAKEN_IN_MS: 20.5,
    }
    malformed_phase = MagicMock(spec=PhaseMessage)
    malformed_phase.usage = {
        INPUT_TOKEN: "30",
        OUTPUT_TOKEN: float("nan"),
        QUERY_TIME_TAKEN_IN_MS: -1,
    }
    missing_usage_phase = MagicMock(spec=PhaseMessage)
    missing_usage_phase.usage = None

    workflow_message = WorkflowMessage("test_workflow")
    workflow_message._phase_messages = [
        valid_phase,
        malformed_phase,
        missing_usage_phase,
    ]

    assert workflow_message.get_total_usage() == {
        "total_input_tokens": 10,
        "total_output_tokens": 0,
        "total_query_time_taken_in_ms": 20.5,
    }


def test_to_log_dict(mocker):
    """
    Test that trajectory logs omit workflow usage and resource metadata.
    """
    mock_phase = mocker.patch.object(
        WorkflowMessage, "phase_messages", new_callable=PropertyMock, return_value=[]
    )
    mock_metadata = mocker.patch.object(
        WorkflowMessage, "metadata_dict", return_value={"key": "value"}
    )
    workflow_message = WorkflowMessage("test_workflow")
    workflow_message.resources_used = {"model": {"config": {"model": "test"}}}
    workflow_message.additional_metadata = {
        "max_input_tokens": 100,
        "nested": {"output_tokens": 20, "kept": "value"},
    }
    log_dict = workflow_message.to_log_dict()

    assert "workflow_usage" not in log_dict
    assert "resources_used" not in log_dict
    assert log_dict["additional_metadata"] == {"nested": {"kept": "value"}}
