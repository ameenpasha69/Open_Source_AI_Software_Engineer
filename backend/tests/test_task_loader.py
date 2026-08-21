from pathlib import Path

from app.evaluation.task_loader import load_tasks

_REAL_TASKS_DIR = Path(__file__).resolve().parents[2] / "evals" / "tasks"


def test_loads_all_real_eval_tasks():
    tasks = load_tasks(_REAL_TASKS_DIR)
    assert len(tasks) == 4
    ids = {t.id for t in tasks}
    assert ids == {
        "task_001_calc_accumulate",
        "task_002_discount_threshold",
        "task_003_inventory_lookup",
        "task_004_username_slug",
    }
    for task in tasks:
        assert task.repo_dir.is_dir()
        assert task.description
        assert task.expected_file is not None


def test_loads_tasks_from_a_fixture_directory(tmp_path):
    task_dir = tmp_path / "task_a"
    (task_dir / "repo").mkdir(parents=True)
    (task_dir / "task.json").write_text('{"id": "task_a", "description": "fix it"}')

    tasks = load_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].id == "task_a"
    assert tasks[0].expected_file is None


def test_ignores_directories_without_task_json(tmp_path):
    (tmp_path / "not_a_task").mkdir()
    (tmp_path / "not_a_task" / "readme.txt").write_text("hi")

    assert load_tasks(tmp_path) == []


def test_tasks_are_sorted_by_directory_name(tmp_path):
    for name in ("task_b", "task_a", "task_c"):
        d = tmp_path / name
        d.mkdir()
        (d / "task.json").write_text(f'{{"id": "{name}", "description": "d"}}')

    tasks = load_tasks(tmp_path)
    assert [t.id for t in tasks] == ["task_a", "task_b", "task_c"]
