import threading
import time
from concurrent.futures import ThreadPoolExecutor

from resources.repo_setup_resource import (
    REPO_SETUP_CONCURRENCY_ENV,
    REPO_SETUP_GATE_DIR_ENV,
    repository_setup_slot,
)


def test_repository_setup_gate_limits_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv(REPO_SETUP_GATE_DIR_ENV, str(tmp_path / "setup-gate"))
    monkeypatch.setenv(REPO_SETUP_CONCURRENCY_ENV, "2")
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def use_setup_slot(number):
        nonlocal active, maximum_active
        with repository_setup_slot(f"repo-{number}"):
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(use_setup_slot, range(6)))

    assert maximum_active == 2


def test_repository_setup_gate_is_disabled_without_matrix_environment(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(REPO_SETUP_GATE_DIR_ENV, raising=False)
    monkeypatch.delenv(REPO_SETUP_CONCURRENCY_ENV, raising=False)

    with repository_setup_slot("direct-run"):
        pass

    assert not (tmp_path / "setup-gate").exists()
