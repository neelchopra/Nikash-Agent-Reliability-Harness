"""Per-trial sandbox: fresh host work dir bind-mounted into a docker container
that runs the MCP server as the stdio transport subprocess."""

import shutil
import uuid
from pathlib import Path

from mcp import StdioServerParameters

from arh.models import Task


class TrialSandbox:
    def __init__(self, task: Task, runs_root: Path):
        self.task = task
        self.trial_id = uuid.uuid4().hex[:12]
        self.work_dir = (runs_root / self.trial_id / "work").resolve()

    def seed(self) -> None:
        """Copy the task's initial_state into a fresh work dir. Never reuse a dir."""
        shutil.copytree(self.task.initial_state, self.work_dir)

    def server_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command="docker",
            args=[
                "run", "-i", "--rm",
                "--network", self.task.network,
                "--memory", self.task.mem_limit,
                "--cpus", "1",
                "-v", f"{self.work_dir}:/work",
                self.task.docker_image,
                *self.task.server_args,
            ],
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.work_dir.parent, ignore_errors=True)
