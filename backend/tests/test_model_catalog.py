from app.llm.catalog import (
    CHAT_MODELS,
    EMBEDDING_MODELS,
    MODEL_CATALOG,
    find_catalog_entry,
    infer_role,
    names_match,
)


def test_catalog_names_are_unique():
    names = [entry.name for entry in MODEL_CATALOG]
    assert len(names) == len(set(names))


def test_catalog_has_exactly_one_recommended_model_per_role():
    for role, entries in (("chat", CHAT_MODELS), ("embedding", EMBEDDING_MODELS)):
        recommended = [e.name for e in entries if e.recommended]
        assert len(recommended) == 1, f"{role} should have one recommended model, got {recommended}"


def test_every_embedding_entry_declares_its_dimensions():
    # Dimensions are what make two embedding models incompatible — the UI
    # warns with this number, so it can't be missing.
    assert all(entry.dimensions for entry in EMBEDDING_MODELS)


def test_chat_entries_do_not_declare_dimensions():
    assert all(entry.dimensions is None for entry in CHAT_MODELS)


def test_find_catalog_entry_tolerates_latest_suffix():
    assert find_catalog_entry("nomic-embed-text") is not None
    assert find_catalog_entry("nomic-embed-text:latest") is find_catalog_entry("nomic-embed-text")


def test_find_catalog_entry_returns_none_for_unknown_model():
    assert find_catalog_entry("some-private-model:v3") is None


def test_infer_role_uses_the_catalog_first():
    assert infer_role("qwen2.5-coder:7b") == "chat"
    assert infer_role("nomic-embed-text:latest") == "embedding"


def test_infer_role_detects_uncatalogued_embedding_models_by_name():
    assert infer_role("granite-embedding:278m") == "embedding"
    assert infer_role("bge-large:335m") == "embedding"


def test_infer_role_defaults_uncatalogued_models_to_chat():
    assert infer_role("my-finetune:latest", family="llama") == "chat"


def test_names_match_ignores_an_implicit_latest_tag():
    assert names_match("mistral:latest", "mistral")
    assert not names_match("mistral:7b", "mistral:latest")
