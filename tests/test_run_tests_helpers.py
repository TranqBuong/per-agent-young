"""Tests for /run-tests helper functions — no subprocess calls needed."""
import pytest
from backend.main import _inject_base_url, _parse_pytest_output


# ── _inject_base_url ──────────────────────────────────────────────────────────

class TestInjectBaseUrl:
    def test_replaces_existing_double_quoted_base_url(self):
        code = 'BASE_URL = "http://old.example.com"'
        result = _inject_base_url(code, "http://new.example.com")
        assert 'BASE_URL = "http://new.example.com"' in result
        assert "http://old.example.com" not in result

    def test_replaces_existing_single_quoted_base_url(self):
        code = "BASE_URL = 'http://old.example.com'"
        result = _inject_base_url(code, "http://new.example.com")
        assert "http://new.example.com" in result
        assert "http://old.example.com" not in result

    def test_replaces_base_url_with_extra_spaces(self):
        code = "BASE_URL  =  'http://old.example.com'"
        result = _inject_base_url(code, "http://new.example.com")
        assert "http://new.example.com" in result

    def test_prepends_when_no_base_url_in_code(self):
        code = "def test_something():\n    pass"
        result = _inject_base_url(code, "http://localhost:8080")
        assert result.startswith('BASE_URL = "http://localhost:8080"')
        assert "def test_something" in result

    def test_original_code_preserved_when_prepending(self):
        code = "import requests\n\ndef test_a(): pass"
        result = _inject_base_url(code, "http://localhost:8080")
        assert "import requests" in result
        assert "def test_a" in result

    def test_replaces_all_occurrences(self):
        code = 'BASE_URL = "http://a.com"\nBASE_URL = "http://b.com"'
        result = _inject_base_url(code, "http://new.com")
        assert result.count("http://new.com") == 2
        assert "http://a.com" not in result
        assert "http://b.com" not in result

    def test_base_url_in_middle_of_code(self):
        code = "import requests\n\nBASE_URL = 'http://old.com'\n\ndef test_a():\n    pass"
        result = _inject_base_url(code, "http://new.com")
        assert 'BASE_URL = "http://new.com"' in result
        assert "def test_a" in result

    def test_https_url_injected_correctly(self):
        code = "BASE_URL = 'http://localhost'"
        result = _inject_base_url(code, "https://api.example.com")
        assert "https://api.example.com" in result

    def test_url_with_path_injected_correctly(self):
        code = "BASE_URL = 'http://old.com'"
        result = _inject_base_url(code, "http://new.com/api/v1")
        assert "http://new.com/api/v1" in result

    def test_empty_base_url_still_replaces(self):
        # Caller is responsible for checking empty; function replaces with whatever is given
        code = 'BASE_URL = "http://old.com"'
        result = _inject_base_url(code, "")
        assert 'BASE_URL = ""' in result


# ── _parse_pytest_output ──────────────────────────────────────────────────────

class TestParsePytestOutput:
    def test_parses_all_passed(self):
        output = "tests/test_a.py ..\n\n2 passed in 0.12s"
        result = _parse_pytest_output(output)
        assert result["passed"] == 2
        assert result["failed"] == 0
        assert result["errors"] == 0
        assert result["total"] == 2

    def test_parses_mixed_passed_failed(self):
        output = "tests/test_a.py F.\n\n1 failed, 1 passed in 0.15s"
        result = _parse_pytest_output(output)
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["total"] == 2

    def test_parses_errors(self):
        output = "ERROR collecting tests/test_a.py\n\n1 error in 0.05s"
        result = _parse_pytest_output(output)
        assert result["errors"] == 1

    def test_all_zero_on_empty_output(self):
        result = _parse_pytest_output("")
        assert result["passed"] == 0
        assert result["failed"] == 0
        assert result["errors"] == 0
        assert result["total"] == 0
        assert result["test_results"] == []

    def test_extracts_passed_test_lines(self):
        output = (
            "PASSED tests/test_a.py::TestSuite::test_login\n"
            "FAILED tests/test_a.py::TestSuite::test_invalid\n"
            "1 failed, 1 passed in 1.2s"
        )
        result = _parse_pytest_output(output)
        assert any("test_login" in t for t in result["test_results"])
        assert any("test_invalid" in t for t in result["test_results"])

    def test_test_results_only_contains_pass_fail_error_lines(self):
        output = (
            "platform linux -- Python 3.11\n"
            "collecting ... done\n"
            "PASSED tests/test_a.py::test_ok\n"
            "some random line\n"
            "1 passed in 0.5s"
        )
        result = _parse_pytest_output(output)
        assert all(
            t.startswith("PASSED") or t.startswith("FAILED") or t.startswith("ERROR")
            for t in result["test_results"]
        )

    def test_total_equals_passed_plus_failed_plus_errors(self):
        output = "3 passed, 1 failed in 0.5s"
        result = _parse_pytest_output(output)
        assert result["total"] == result["passed"] + result["failed"] + result["errors"]

    def test_only_failed_no_passed(self):
        output = "FAILED tests/test_a.py::test_bad\n\n1 failed in 0.3s"
        result = _parse_pytest_output(output)
        assert result["failed"] == 1
        assert result["passed"] == 0

    def test_large_counts_parsed_correctly(self):
        output = "100 passed, 25 failed, 3 errors in 12.5s"
        result = _parse_pytest_output(output)
        assert result["passed"] == 100
        assert result["failed"] == 25
        assert result["errors"] == 3

    def test_no_test_results_when_no_verbose_lines(self):
        output = "2 passed in 0.12s"
        result = _parse_pytest_output(output)
        assert result["test_results"] == []

    def test_only_passed_count_present(self):
        output = "all good\n5 passed in 1.0s"
        result = _parse_pytest_output(output)
        assert result["passed"] == 5
        assert result["failed"] == 0

    def test_handles_error_in_word_errors(self):
        output = "2 errors in 0.1s"
        result = _parse_pytest_output(output)
        assert result["errors"] == 2
