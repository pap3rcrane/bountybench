import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from resources.base_resource import BaseResourceConfig
from resources.base_setup_resource import BaseSetupResource
from utils.logger import get_main_logger

logger = get_main_logger(__name__)

REPO_SETUP_GATE_DIR_ENV = "BOUNTYBENCH_REPO_SETUP_GATE_DIR"
REPO_SETUP_CONCURRENCY_ENV = "BOUNTYBENCH_REPO_SETUP_CONCURRENCY"


@contextmanager
def repository_setup_slot(resource_id: str) -> Iterator[None]:
    """Limit concurrent repository setup scripts across matrix workers."""
    gate_dir_value = os.environ.get(REPO_SETUP_GATE_DIR_ENV)
    concurrency_value = os.environ.get(REPO_SETUP_CONCURRENCY_ENV)
    if not gate_dir_value or not concurrency_value:
        yield
        return

    try:
        concurrency = int(concurrency_value)
    except ValueError as error:
        raise RuntimeError(
            f"{REPO_SETUP_CONCURRENCY_ENV} must be a positive integer"
        ) from error
    if concurrency <= 0:
        raise RuntimeError(
            f"{REPO_SETUP_CONCURRENCY_ENV} must be a positive integer"
        )

    gate_dir = Path(gate_dir_value)
    gate_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Waiting for repository setup slot for {resource_id} "
        f"(limit: {concurrency})"
    )

    while True:
        for slot_number in range(1, concurrency + 1):
            lock_file: TextIO = (gate_dir / f"slot-{slot_number}.lock").open("a+")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_file.close()
                continue

            logger.info(
                f"Acquired repository setup slot {slot_number}/{concurrency} "
                f"for {resource_id}"
            )
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                logger.info(
                    f"Released repository setup slot {slot_number}/{concurrency} "
                    f"for {resource_id}"
                )
            return
        time.sleep(0.25)


@dataclass
class RepoSetupResourceConfig(BaseResourceConfig):
    """Configuration for RepoSetupResource"""

    task_dir: Path

    def validate(self) -> None:
        """Validate Repo Setup configuration"""
        if not self.task_dir.exists():
            raise ValueError(f"Invalid task_dir: {self.task_dir}")


class RepoSetupResource(BaseSetupResource):
    """RepoSetupResource for initializing and managing task-level containers."""

    def __init__(self, resource_id: str, config: RepoSetupResourceConfig):
        # Call the superclass constructor first
        super().__init__(resource_id, config)

        # Set required properties
        self.task_dir = self._resource_config.task_dir
        self.setup_script_name = "setup_repo_env.sh"

        # Set work_dir for task setup (directly the task directory)
        self.work_dir = self.task_dir

        # Run the setup process
        self.setup()

    def _start(self) -> None:
        with repository_setup_slot(self.resource_id):
            super()._start()

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "RepoSetupResource":
        """
        Creates a RepoSetupResource instance from a serialized dictionary.
        """
        common_attrs = super().from_dict(data, **kwargs)
        
        config = RepoSetupResourceConfig(
            task_dir=Path(data["task_dir"]),
        )
        
        instance = cls(common_attrs["resource_id"], config)
        
        instance.container_names = common_attrs["container_names"]
        
        return instance
