import re

_FAILED_LINE = re.compile(r"^FAILED (\S+)", re.MULTILINE)
_SUMMARY_COUNT = re.compile(r"(\d+) (passed|failed|error|skipped)")

# pytest exit codes: 0 all passed, 1 tests failed, 2 interrupted, 3 internal
# error, 4 usage error, 5 no tests collected. 2/3/4/5 all mean pytest itself
# couldn't run the suite as asked — that's an environment/config problem,
# not "the code under test is wrong."
_ENVIRONMENT_EXIT_CODES = frozenset({2, 3, 4, 5})


def parse_pytest_output(stdout: str) -> tuple[int | None, int | None, list[str]]:
    """Returns (total_tests, passed_tests, failed_test_ids) parsed from
    pytest's default terminal output — the summary line (e.g.
    "3 failed, 5 passed in 1.23s") and the "FAILED <nodeid>" lines pytest
    prints by default whenever any test fails, no extra flags required."""
    failed_tests = _FAILED_LINE.findall(stdout)
    counts = {label: int(n) for n, label in _SUMMARY_COUNT.findall(stdout)}

    if not counts:
        return None, None, failed_tests

    passed = counts.get("passed")
    total = sum(counts.values())
    return total, passed, failed_tests


def classify_failure(exit_code: int, timed_out: bool, stdout: str, stderr: str) -> str:
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return "none"

    combined = stdout + stderr
    if "SyntaxError" in combined:
        return "syntax_error"
    if "ModuleNotFoundError" in combined or "ImportError" in combined:
        return "dependency_error"
    if exit_code in _ENVIRONMENT_EXIT_CODES:
        return "environment_error"
    return "test_failure"
