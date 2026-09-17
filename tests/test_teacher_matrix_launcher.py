import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "run_teacher_matrix.sh"
DOCKERD_ENTRYPOINT = PROJECT_ROOT / "tools" / "dockerd-entrypoint.sh"


def _argument_values(arguments, option):
    return [
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == option
    ]


def test_all_modes_reuse_one_pool_of_thirty_isolated_workers(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    python_log_dir = tmp_path / "python_logs"
    python_log_dir.mkdir()

    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n" 'printf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n' "exit 0\n"
    )
    fake_docker.chmod(0o755)

    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'mode="unknown"\n'
        'previous=""\n'
        'for argument in "$@"; do\n'
        '  if [[ "$previous" == "--teacher-mode" ]]; then mode="$argument"; fi\n'
        '  previous="$argument"\n'
        "done\n"
        'printf "%s\\n" "$*" > "$FAKE_PYTHON_LOG_DIR/$mode.log"\n'
        "exit 0\n"
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PYTHON_EXECUTABLE": str(fake_python),
            "FAKE_DOCKER_LOG": str(docker_log),
            "FAKE_PYTHON_LOG_DIR": str(python_log_dir),
            "BATCH_LOG_ROOT": str(tmp_path / "batch_logs"),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "MATRIX_WORKER_LOCK_DIRECTORY": str(tmp_path / "worker.lock"),
            "BOUNTY_AGENT_IMAGE": "example/bountyagent:amd64",
        }
    )

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--jobs",
            "all",
            "--phase-iterations",
            "17",
            "--no-progress",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    docker_commands = docker_log.read_text().splitlines()
    detached_run_commands = [
        shlex.split(command)
        for command in docker_commands
        if command.startswith("run -d ")
    ]
    worker_commands = [
        arguments
        for arguments in detached_run_commands
        if _argument_values(arguments, "--name")[0].startswith("backend-worker-")
    ]
    cache_commands = [
        arguments
        for arguments in detached_run_commands
        if _argument_values(arguments, "--name")
        == ["bountybench-matrix-registry-cache"]
    ]
    assert len(worker_commands) == 30
    assert len(cache_commands) == 1

    cache_arguments = cache_commands[0]
    cache_network = _argument_values(cache_arguments, "--network")[0]
    assert cache_network != "none"
    assert _argument_values(cache_arguments, "--network-alias") == [
        "matrix-registry-cache"
    ]
    assert _argument_values(cache_arguments, "--env") == [
        "REGISTRY_PROXY_REMOTEURL=https://registry-1.docker.io"
    ]
    assert _argument_values(cache_arguments, "--volume") == [
        "bountybench-matrix-registry-cache:/var/lib/registry"
    ]
    assert cache_arguments[-1] == "mirror.gcr.io/library/registry:2"

    cache_network_commands = [
        shlex.split(command)
        for command in docker_commands
        if command.startswith("network connect ")
    ]
    assert len(cache_network_commands) == 29
    assert {
        _argument_values(arguments, "--alias")[0]
        for arguments in cache_network_commands
    } == {"matrix-registry-cache"}
    assert {
        arguments[-1] for arguments in cache_network_commands
    } == {"bountybench-matrix-registry-cache"}
    connected_cache_networks = {
        arguments[-2] for arguments in cache_network_commands
    }

    network_create_commands = [
        shlex.split(command)
        for command in docker_commands
        if command.startswith("network create ")
    ]
    assert len(network_create_commands) == 30
    assert {
        tuple(_argument_values(arguments, "--subnet"))
        for arguments in network_create_commands
    } == {("0.0.0.0/24",)}
    created_worker_networks = {
        arguments[-1] for arguments in network_create_commands
    }

    worker_names = set()
    worker_networks = set()
    dind_volumes = set()
    artifact_roots = set()
    for arguments in worker_commands:
        worker_name = _argument_values(arguments, "--name")[0]
        worker_names.add(worker_name)
        worker_networks.add(_argument_values(arguments, "--network")[0])
        assert "DOCKER_REGISTRY_MIRROR=http://matrix-registry-cache:5000" in (
            _argument_values(arguments, "--env")
        )
        assert "--memory" not in arguments
        assert "--memory-swap" not in arguments

        volumes = _argument_values(arguments, "--volume")
        dind_mount = next(
            volume for volume in volumes if volume.endswith(":/var/lib/docker")
        )
        dind_volumes.add(dind_mount.split(":", 1)[0])
        assert any(
            volume.endswith(":/matrix-setup-gate") for volume in volumes
        )
        assert "BOUNTYBENCH_REPO_SETUP_GATE_DIR=/matrix-setup-gate" in (
            _argument_values(arguments, "--env")
        )
        assert "BOUNTYBENCH_REPO_SETUP_CONCURRENCY=5" in (
            _argument_values(arguments, "--env")
        )
        assert "BOUNTYBENCH_DISABLE_KALI_DOCKER=1" in (
            _argument_values(arguments, "--env")
        )
        assert "BOUNTY_AGENT_IMAGE" in _argument_values(arguments, "--env")
        log_mount = next(volume for volume in volumes if volume.endswith(":/app/logs"))
        artifact_roots.add(str(Path(log_mount.split(":", 1)[0]).parent))
        assert worker_name in log_mount

    expected_workers = {f"backend-worker-{number}" for number in range(1, 31)}
    assert worker_names == expected_workers
    assert len(worker_networks) == 30
    assert created_worker_networks == worker_networks
    assert cache_network in worker_networks
    assert connected_cache_networks == worker_networks - {cache_network}
    assert len(dind_volumes) == 30
    assert len(artifact_roots) == 30

    mode_commands = [
        shlex.split(path.read_text()) for path in sorted(python_log_dir.glob("*.log"))
    ]
    assert len(mode_commands) == 3
    workers_by_mode = {}
    for arguments in mode_commands:
        mode = _argument_values(arguments, "--teacher-mode")[0]
        assert _argument_values(arguments, "--jobs") == ["30"]
        assert _argument_values(arguments, "--setup-jobs") == ["5"]
        assert _argument_values(arguments, "--teacher-max-input-tokens") == [
            "1048576"
        ]
        assert _argument_values(arguments, "--teacher-max-output-tokens") == [
            "65536"
        ]
        assert _argument_values(arguments, "--phase-iterations") == ["17"]
        workers_by_mode[mode] = set(_argument_values(arguments, "--backend-container"))

    assert workers_by_mode == {
        "observe": expected_workers,
        "steer": expected_workers,
        "objective_rewrite": expected_workers,
    }


def test_worker_entrypoint_uses_optional_mirror_without_architecture_specific_helper():
    entrypoint = DOCKERD_ENTRYPOINT.read_text()

    assert '--registry-mirror "$DOCKER_REGISTRY_MIRROR"' in entrypoint
    assert '--insecure-registry "$MIRROR_HOST"' in entrypoint
    assert "docker login --username" in entrypoint
    assert '"${BOUNTY_AGENT_IMAGE:-}" == *.pkg.dev/*' in entrypoint
    assert "metadata.google.internal" in entrypoint
    assert "--username oauth2accesstoken" in entrypoint
    assert "docker-credential-pass" not in entrypoint
    assert "linux-arm64" not in entrypoint


def test_only_prompt_forwards_one_prompt_and_omits_baseline(tmp_path):
    fake_python = tmp_path / "python"
    python_log = tmp_path / "python.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" > "$FAKE_PYTHON_LOG"\n'
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_EXECUTABLE": str(fake_python),
            "FAKE_PYTHON_LOG": str(python_log),
            "BATCH_LOG_ROOT": str(tmp_path / "batch_logs"),
            "RUNS_ROOT": str(tmp_path / "runs"),
        }
    )

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--modes",
            "steer",
            "--jobs",
            "1",
            "--prompt-placement",
            "system",
            "--only-prompt",
            "comparative_ranking",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(python_log.read_text())
    assert _argument_values(arguments, "--prompt-file") == [
        "prompts/system_prompts/comparative_ranking.txt"
    ]
    assert "--exclude-baseline" in arguments


def test_launcher_stops_immediately_when_a_worker_network_cannot_be_created(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n'
        'if [[ "$*" == network\\ create*worker-2 ]]; then exit 1; fi\n'
        "exit 0\n"
    )
    fake_docker.chmod(0o755)

    lock_directory = tmp_path / "worker.lock"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
            "BATCH_LOG_ROOT": str(tmp_path / "batch_logs"),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "MATRIX_WORKER_LOCK_DIRECTORY": str(lock_directory),
        }
    )

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--modes",
            "observe",
            "--jobs",
            "2",
            "--no-progress",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "failed to create isolated network" in result.stderr
    assert not lock_directory.exists()
    docker_commands = docker_log.read_text().splitlines()
    assert not any(command.startswith("run -d ") for command in docker_commands)
