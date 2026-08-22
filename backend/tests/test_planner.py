from app.agents.planner import _FALLBACK_PLAN, Planner

from tests.conftest import FakeLLMProvider


async def test_planner_parses_valid_plan():
    llm = FakeLLMProvider(responses=['{"steps": ["Search for the bug", "Read the file", "Diagnose it"]}'])
    planner = Planner(llm)

    plan = await planner.create_plan("Orders endpoint 500s", "orders-service")

    assert plan == ["Search for the bug", "Read the file", "Diagnose it"]
    assert llm.json_mode_calls == [True]


async def test_planner_falls_back_on_malformed_json():
    llm = FakeLLMProvider(responses=["not valid json at all"])
    planner = Planner(llm)

    plan = await planner.create_plan("Orders endpoint 500s", "orders-service")

    assert plan == _FALLBACK_PLAN


async def test_planner_falls_back_on_wrong_shape():
    llm = FakeLLMProvider(responses=['{"not_steps": "wrong key"}'])
    planner = Planner(llm)

    plan = await planner.create_plan("task", "repo")

    assert plan == _FALLBACK_PLAN


async def test_planner_filters_blank_steps():
    llm = FakeLLMProvider(responses=['{"steps": ["Step one", "  ", "", "Step two"]}'])
    planner = Planner(llm)

    plan = await planner.create_plan("task", "repo")

    assert plan == ["Step one", "Step two"]


async def test_planner_prompt_covers_non_bug_tasks_too():
    # Regression: the prompt used to say "given a bug report," exclusively —
    # for a task like "make this a scientific calculator and rename the
    # file," the planner produced a generic SDLC plan (branch, design doc,
    # peer review) with no step the agent could actually execute, and the
    # agent looped re-doing step 1 forever since nothing else was actionable.
    llm = FakeLLMProvider(responses=['{"steps": ["step"]}'])
    planner = Planner(llm)

    await planner.create_plan("task", "repo")

    system_prompt = llm.received_messages[0][0].content
    assert "bug" in system_prompt.lower()  # still covers bug reports...
    assert "added, changed, renamed" in system_prompt.lower() or "renamed" in system_prompt.lower()


async def test_planner_prompt_forbids_steps_with_no_real_capability():
    llm = FakeLLMProvider(responses=['{"steps": ["step"]}'])
    planner = Planner(llm)

    await planner.create_plan("task", "repo")

    system_prompt = llm.received_messages[0][0].content.lower()
    assert "branch" in system_prompt
    assert "peer review" in system_prompt


async def test_fallback_plan_does_not_assume_a_bug_report():
    # The fallback plan used to end on "form a hypothesis about the root
    # cause" — meaningless for a task that isn't a bug at all.
    assert not any("root cause" in step.lower() for step in _FALLBACK_PLAN)
