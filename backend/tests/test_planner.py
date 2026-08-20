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
