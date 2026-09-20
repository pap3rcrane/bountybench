import ast
from pathlib import Path

import pytest


AGENT_FILES = (
    "agents/exploit_agent/exploit_agent.py",
    "agents/detect_agent/detect_agent.py",
    "agents/detect_patch_agent/detect_patch_agent.py",
    "agents/patch_agent/patch_agent.py",
)


@pytest.mark.parametrize("relative_path", AGENT_FILES)
def test_bounty_agent_container_image_uses_environment(relative_path):
    source_path = Path(__file__).parents[2] / relative_path
    tree = ast.parse(source_path.read_text())
    docker_image_values = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "DockerActionMessage":
            continue
        docker_image_values.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == "docker_image"
        )

    assert docker_image_values
    for value in docker_image_values:
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Attribute)
        assert isinstance(value.func.value, ast.Name)
        assert value.func.value.id == "os"
        assert value.func.attr == "getenv"
        assert [argument.value for argument in value.args] == [
            "BOUNTY_AGENT_IMAGE",
            "cybench/bountyagent:latest",
        ]
