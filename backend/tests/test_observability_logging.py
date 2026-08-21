import json
import logging

import pytest
from app.observability.logging import (
    JsonFormatter,
    TextFormatter,
    _ContextFilter,
    configure_logging,
    request_id_var,
    run_id_var,
)


@pytest.fixture(autouse=True)
def _restore_root_logger_state():
    """configure_logging() mutates the *global* root logger — restore its
    handlers/level afterward so these tests don't leak state (e.g. a
    StreamHandler bound to a capsys buffer that pytest later closes) into
    unrelated tests running later in the same session."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


def _make_record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_correlation_ids_when_present():
    record = _make_record(request_id="req-123", run_id="run-456")

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "req-123"
    assert payload["run_id"] == "run-456"


def test_json_formatter_omits_correlation_ids_when_absent():
    record = _make_record(request_id=None, run_id=None)

    payload = json.loads(JsonFormatter().format(record))

    assert "request_id" not in payload
    assert "run_id" not in payload


def test_json_formatter_includes_arbitrary_extra_fields():
    record = _make_record(request_id=None, run_id=None, tool_name="run_tests", duration_seconds=1.23)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["tool_name"] == "run_tests"
    assert payload["duration_seconds"] == 1.23


def test_json_formatter_output_is_a_single_line():
    record = _make_record(request_id=None, run_id=None)

    output = JsonFormatter().format(record)

    assert "\n" not in output


def test_text_formatter_appends_correlation_ids_when_present():
    record = _make_record(request_id="req-123", run_id=None)

    output = TextFormatter("%(message)s").format(record)

    assert "hello" in output
    assert "request_id=req-123" in output


def test_text_formatter_has_no_bracket_suffix_without_correlation_ids():
    record = _make_record(request_id=None, run_id=None)

    output = TextFormatter("%(message)s").format(record)

    assert output == "hello"


def test_request_id_and_run_id_context_vars_default_to_none():
    assert request_id_var.get() is None
    assert run_id_var.get() is None


def test_context_filter_injects_current_contextvar_values():
    token_request = request_id_var.set("req-789")
    token_run = run_id_var.set("run-abc")
    try:
        record = logging.LogRecord(
            name="app.test", level=logging.INFO, pathname=__file__, lineno=1, msg="x", args=(), exc_info=None
        )
        assert _ContextFilter().filter(record) is True
        assert record.request_id == "req-789"
        assert record.run_id == "run-abc"
    finally:
        request_id_var.reset(token_request)
        run_id_var.reset(token_run)


def test_configure_logging_json_produces_parseable_output(capsys):
    configure_logging("INFO", "json")
    logging.getLogger("app.test").info("configured ok")

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["message"] == "configured ok"


def test_configure_logging_replaces_prior_handlers():
    configure_logging("INFO", "text")
    handler_count_first = len(logging.getLogger().handlers)
    configure_logging("INFO", "text")

    assert len(logging.getLogger().handlers) == handler_count_first
