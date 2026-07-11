from pathlib import Path

import yaml

from arh.models import Task


def load_task(path: Path) -> Task:
    """Load a task YAML and resolve initial_state relative to the YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = Task.model_validate(data)
    task.initial_state = (path.parent / task.initial_state).resolve()
    return task
