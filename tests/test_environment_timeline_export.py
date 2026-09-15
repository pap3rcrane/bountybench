from datetime import datetime, timezone

from environment_timelines.export_latest_agent_timeline import automatic_output_path


def test_automatic_output_path_includes_workflow_name():
    data = {
        "workflow_metadata": {
            "workflow_name": "DetectWorkflow",
            "task": {
                "task_dir": "bountytasks/lunary",
                "bounty_number": "2",
            },
        }
    }
    generated_at = datetime(2026, 8, 20, 13, 14, 15, 123456, timezone.utc)

    output = automatic_output_path(data, generated_at)

    assert output.name == (
        "lunary_bounty_2_DetectWorkflow_"
        "2026-08-20_13-14-15-123456+0000_environment_timeline.json"
    )
