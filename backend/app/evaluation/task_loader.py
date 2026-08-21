import json
from pathlib import Path

from app.evaluation.models import TaskDefinition


def load_tasks(tasks_dir: Path) -> list[TaskDefinition]:
    """Each task is a directory under `tasks_dir` containing `task.json`
    (description, optional expected_file) and a `repo/` fixture — the
    "repository state" the agent investigates. Loaded in sorted directory
    order so eval runs are reproducible."""
    tasks = []
    for task_dir in sorted(tasks_dir.iterdir()):
        task_json = task_dir / "task.json"
        if not task_json.is_file():
            continue
        data = json.loads(task_json.read_text())
        tasks.append(
            TaskDefinition(
                id=data["id"],
                description=data["description"],
                expected_file=data.get("expected_file"),
                repo_dir=task_dir / "repo",
            )
        )
    return tasks
