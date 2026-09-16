import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "run_teacher_matrix.sh"


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
        }
    )

    result = subprocess.run(
        [str(LAUNCHER), "--jobs", "all", "--no-progress"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    docker_commands = docker_log.read_text().splitlines()
    worker_commands = [
        shlex.split(command)
        for command in docker_commands
        if command.startswith("run -d ")
    ]
    assert len(worker_commands) == 30

    worker_names = set()
    worker_networks = set()
    dind_volumes = set()
    artifact_roots = set()
    for arguments in worker_commands:
        worker_name = _argument_values(arguments, "--name")[0]
        worker_names.add(worker_name)
        worker_networks.add(_argument_values(arguments, "--network")[0])
        assert "--memory" not in arguments
        assert "--memory-swap" not in arguments

        volumes = _argument_values(arguments, "--volume")
        dind_mount = next(
            volume for volume in volumes if volume.endswith(":/var/lib/docker")
        )
        dind_volumes.add(dind_mount.split(":", 1)[0])
        log_mount = next(volume for volume in volumes if volume.endswith(":/app/logs"))
        artifact_roots.add(str(Path(log_mount.split(":", 1)[0]).parent))
        assert worker_name in log_mount

    expected_workers = {f"backend-worker-{number}" for number in range(1, 31)}
    assert worker_names == expected_workers
    assert len(worker_networks) == 30
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
        workers_by_mode[mode] = set(_argument_values(arguments, "--backend-container"))

    assert workers_by_mode == {
        "observe": expected_workers,
        "steer": expected_workers,
        "objective_rewrite": expected_workers,
    }
