from app.tools.test_result_parser import classify_failure, parse_pytest_output

_ALL_PASS_OUTPUT = """\
============================= test session starts ==============================
collected 2 items

test_sample.py ..                                                        [100%]

============================== 2 passed in 0.01s ==============================
"""

_ONE_FAIL_OUTPUT = """\
============================= test session starts ==============================
collected 2 items

test_sample.py .F                                                        [100%]

=================================== FAILURES ===================================
__________________________________ test_fail ___________________________________

    def test_fail():
>       assert 1 == 2
E       assert 1 == 2

test_sample.py:5: AssertionError
=========================== short test summary info ============================
FAILED test_sample.py::test_fail - assert 1 == 2
========================= 1 failed, 1 passed in 0.01s ==========================
"""

_MULTI_FAIL_OUTPUT = """\
FAILED test_a.py::test_one - AssertionError
FAILED test_a.py::test_two - ValueError
========================= 2 failed, 3 passed in 0.05s ==========================
"""


def test_parse_pytest_output_all_passed():
    total, passed, failed = parse_pytest_output(_ALL_PASS_OUTPUT)
    assert total == 2
    assert passed == 2
    assert failed == []


def test_parse_pytest_output_one_failure():
    total, passed, failed = parse_pytest_output(_ONE_FAIL_OUTPUT)
    assert total == 2
    assert passed == 1
    assert failed == ["test_sample.py::test_fail"]


def test_parse_pytest_output_multiple_failures():
    total, passed, failed = parse_pytest_output(_MULTI_FAIL_OUTPUT)
    assert total == 5
    assert passed == 3
    assert failed == ["test_a.py::test_one", "test_a.py::test_two"]


def test_parse_pytest_output_no_recognizable_summary():
    total, passed, failed = parse_pytest_output("some unrelated garbage output")
    assert total is None
    assert passed is None
    assert failed == []


def test_classify_failure_none_on_success():
    assert classify_failure(0, False, "", "") == "none"


def test_classify_failure_timeout_takes_priority():
    assert classify_failure(0, True, "", "") == "timeout"


def test_classify_failure_syntax_error():
    assert classify_failure(2, False, "SyntaxError: invalid syntax", "") == "syntax_error"


def test_classify_failure_dependency_error():
    assert classify_failure(1, False, "", "ModuleNotFoundError: No module named 'foo'") == "dependency_error"
    assert classify_failure(1, False, "", "ImportError: cannot import name 'bar'") == "dependency_error"


def test_classify_failure_environment_error_from_exit_code():
    assert classify_failure(4, False, "usage error", "") == "environment_error"
    assert classify_failure(5, False, "no tests collected", "") == "environment_error"


def test_classify_failure_test_failure_default():
    assert classify_failure(1, False, "1 failed, 1 passed", "") == "test_failure"
