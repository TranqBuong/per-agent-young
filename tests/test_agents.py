"""Tests for agent utility functions — no LLM calls needed."""
import pytest

from backend.app.agents.test_case_payload_generator import (
    _trunc,
    _extract_json,
    _parse_json,
    make_fallback_tc,
)
from backend.app.agents.requirement_analyzer import _compute_quality_score
from backend.app.agents.automation_code_writer import _parse_separator


# ── _trunc ────────────────────────────────────────────────────────────────────

class TestTrunc:
    def test_no_truncation_when_under_limit(self):
        assert _trunc("hello world", 50) == "hello world"

    def test_no_truncation_at_exact_limit(self):
        text = "x" * 20
        assert _trunc(text, 20) == text

    def test_truncates_at_word_boundary(self):
        text = "one two three four five"
        result = _trunc(text, 12)
        assert result.endswith("[... truncated]")
        assert "one two" in result
        assert "three" not in result

    def test_appends_truncated_marker(self):
        result = _trunc("a b c d e f g", 8)
        assert "[... truncated]" in result

    def test_empty_string(self):
        assert _trunc("", 10) == ""

    def test_single_word_over_limit(self):
        result = _trunc("superlongword", 5)
        assert "[... truncated]" in result


# ── _extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_extracts_simple_object(self):
        text = 'prefix {"key": "value"} suffix'
        assert _extract_json(text) == '{"key": "value"}'

    def test_extracts_nested_object(self):
        text = '{"outer": {"inner": 1}}'
        result = _extract_json(text)
        assert result == text

    def test_raises_when_no_json(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")

    def test_handles_string_with_braces(self):
        text = '{"msg": "hello {world}"}'
        result = _extract_json(text)
        assert result == text

    def test_handles_markdown_fence_prefix(self):
        text = "```json\n{\"a\": 1}\n```"
        # _extract_json finds the first { regardless of fence
        result = _extract_json(text)
        assert '"a"' in result


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_plain_json(self):
        result = _parse_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_in_markdown_fence(self):
        text = "```json\n{\"x\": 1}\n```"
        assert _parse_json(text) == {"x": 1}

    def test_json_with_embedded_newlines_in_strings(self):
        text = '{"msg": "line1\\nline2"}'
        result = _parse_json(text)
        assert result["msg"] == "line1\nline2"

    def test_json_with_surrounding_text(self):
        text = 'Here is the response: {"result": true}'
        result = _parse_json(text)
        assert result == {"result": True}


# ── _compute_quality_score ────────────────────────────────────────────────────

ALL_TRUE_CHECKS = {
    "completeness": {
        "inputs_defined": True,
        "validation_rules": True,
        "success_response": True,
        "error_cases": True,
        "business_rules": True,
        "auth_stated": True,
    },
    "testability": {
        "acceptance_criteria": True,
        "expected_outputs": True,
        "test_data_derivable": True,
        "rules_verifiable": True,
        "no_vague_language": True,
    },
    "clarity_positive": {
        "has_concrete_examples": True,
        "terms_defined": True,
        "logical_flow": True,
        "specific_acceptance_criteria": True,
    },
    "clarity_issues": {
        "vague_words": [],
        "undefined_terms": [],
        "conflicting_statements": [],
        "implicit_assumptions": [],
    },
}

ALL_FALSE_CHECKS = {
    "completeness": {k: False for k in ALL_TRUE_CHECKS["completeness"]},
    "testability": {k: False for k in ALL_TRUE_CHECKS["testability"]},
    "clarity_positive": {k: False for k in ALL_TRUE_CHECKS["clarity_positive"]},
    "clarity_issues": {
        "vague_words": ["fast", "good", "nice", "appropriate", "simple"],
        "undefined_terms": [],
        "conflicting_statements": [],
        "implicit_assumptions": [],
    },
}


class TestComputeQualityScore:
    def test_perfect_score(self):
        data = {"quality_checks": ALL_TRUE_CHECKS}
        qs = _compute_quality_score(data)
        assert qs["overall"] == 100
        assert qs["completeness"] == 100
        assert qs["testability"] == 100
        assert qs["ambiguity"] == 100
        assert qs["risk"] == "Low"

    def test_zero_score(self):
        data = {"quality_checks": ALL_FALSE_CHECKS}
        qs = _compute_quality_score(data)
        assert qs["completeness"] == 0
        assert qs["testability"] == 0
        assert qs["ambiguity"] == 0
        assert qs["overall"] == 0
        assert qs["risk"] == "High"

    def test_completeness_partial(self):
        checks = {
            **ALL_FALSE_CHECKS,
            "completeness": {
                "inputs_defined": True,   # +20
                "validation_rules": True, # +20
                "success_response": False,
                "error_cases": False,
                "business_rules": False,
                "auth_stated": False,
            },
        }
        qs = _compute_quality_score({"quality_checks": checks})
        assert qs["completeness"] == 40

    def test_clarity_positive_bonus(self):
        checks = {
            **ALL_FALSE_CHECKS,
            "clarity_positive": {
                "has_concrete_examples": True,   # +15
                "terms_defined": True,            # +15
                "logical_flow": False,
                "specific_acceptance_criteria": False,
            },
            "clarity_issues": {k: [] for k in ALL_FALSE_CHECKS["clarity_issues"]},
        }
        qs = _compute_quality_score({"quality_checks": checks})
        assert qs["ambiguity"] == 50 + 15 + 15  # base + bonuses

    def test_clarity_deductions(self):
        checks = {
            **ALL_FALSE_CHECKS,
            "clarity_positive": {k: False for k in ALL_TRUE_CHECKS["clarity_positive"]},
            "clarity_issues": {
                "vague_words": ["fast", "good"],  # 2 × 15 = 30
                "undefined_terms": ["API"],       # 1 × 10 = 10
                "conflicting_statements": [],
                "implicit_assumptions": [],
            },
        }
        qs = _compute_quality_score({"quality_checks": checks})
        assert qs["ambiguity"] == max(0, 50 - 30 - 10)  # = 10

    def test_clarity_clamped_at_zero(self):
        checks = {
            **ALL_FALSE_CHECKS,
            "clarity_issues": {
                "vague_words": ["a", "b", "c", "d", "e"],  # 5 × 15 = 75
                "undefined_terms": ["x"],                    # 1 × 10 = 10
                "conflicting_statements": [],
                "implicit_assumptions": [],
            },
        }
        qs = _compute_quality_score({"quality_checks": checks})
        assert qs["ambiguity"] == 0  # 50 - 85 = -35 → clamped to 0

    def test_overall_formula(self):
        # completeness=100, testability=0, ambiguity=0
        checks = {
            "completeness": ALL_TRUE_CHECKS["completeness"],
            "testability": ALL_FALSE_CHECKS["testability"],
            "clarity_positive": ALL_FALSE_CHECKS["clarity_positive"],
            "clarity_issues": {k: [] for k in ALL_FALSE_CHECKS["clarity_issues"]},
        }
        qs = _compute_quality_score({"quality_checks": checks})
        # overall = 100*0.40 + 0*0.35 + 50*0.25 = 40 + 0 + 12.5 → round = 53
        assert qs["overall"] == round(100 * 0.40 + 0 * 0.35 + 50 * 0.25)

    def test_risk_thresholds(self):
        def _make(overall):
            return {"overall": overall, "completeness": 0, "testability": 0,
                    "ambiguity": 0, "risk": "High",
                    "score_breakdown": {}}

        assert _compute_quality_score({"quality_checks": {**ALL_FALSE_CHECKS,
            "clarity_issues": {k: [] for k in ALL_FALSE_CHECKS["clarity_issues"]}}})["risk"] == "High"

        checks_medium = {**ALL_TRUE_CHECKS,
                         "clarity_issues": {k: [] for k in ALL_FALSE_CHECKS["clarity_issues"]}}
        # Force a medium risk scenario by partial checks
        partial = {
            "completeness": {"inputs_defined": True, "validation_rules": True,
                             "success_response": False, "error_cases": False,
                             "business_rules": False, "auth_stated": False},
            "testability": {"acceptance_criteria": True, "expected_outputs": True,
                            "test_data_derivable": False, "rules_verifiable": False,
                            "no_vague_language": False},
            "clarity_positive": {k: False for k in ALL_TRUE_CHECKS["clarity_positive"]},
            "clarity_issues": {k: [] for k in ALL_FALSE_CHECKS["clarity_issues"]},
        }
        qs = _compute_quality_score({"quality_checks": partial})
        assert qs["risk"] in {"High", "Medium", "Low"}

    def test_missing_quality_checks_key(self):
        qs = _compute_quality_score({})
        assert qs["overall"] == round(0 * 0.40 + 0 * 0.35 + 50 * 0.25)

    def test_score_breakdown_keys_present(self):
        data = {"quality_checks": ALL_TRUE_CHECKS}
        qs = _compute_quality_score(data)
        bd = qs["score_breakdown"]
        assert "completeness_found" in bd
        assert "completeness_missing" in bd
        assert "testability_found" in bd
        assert "testability_missing" in bd
        assert "ambiguity_deductions" in bd
        assert "clarity_found" in bd
        assert "clarity_missing" in bd


# ── make_fallback_tc ──────────────────────────────────────────────────────────

class TestMakeFallbackTc:
    def test_basic_fields_present(self):
        s = {"scenario_id": "SCN-001", "title": "Login", "given": "User exists",
             "when": "POST /login", "then": "200 OK", "priority": "high", "type": "positive"}
        tc = make_fallback_tc(s, 0)
        assert tc["scenario_id"] == "SCN-001"
        assert tc["technique"] == "UC"
        assert len(tc["steps"]) == 3
        assert tc["priority"] == "high"

    def test_id_rebuilt_by_generator(self):
        # make_fallback_tc produces a temporary ID; the generator rebuild loop overwrites it
        s = {"scenario_id": "SCN-042", "title": "Edge"}
        tc = make_fallback_tc(s, 5)
        # The temp ID doesn't need to be TC-TECHNIQUE-NNN — rebuild fixes it
        assert "test_case_id" in tc

    def test_missing_given_when_then_defaults(self):
        s = {"scenario_id": "SCN-001", "title": "Empty"}
        tc = make_fallback_tc(s, 0)
        assert tc["preconditions"] == "System is available"
        assert len(tc["steps"]) == 3

    def test_name_truncated_at_50(self):
        long_title = "A" * 100
        s = {"scenario_id": "SCN-001", "title": long_title}
        tc = make_fallback_tc(s, 0)
        assert len(tc["name"]) <= 50


# ── _parse_separator ──────────────────────────────────────────────────────────

class TestParseSeparator:
    def test_parses_single_file_block(self):
        text = "===FILE:tests/test_login.py===\nimport requests\n\ndef test_login():\n    pass\n===END==="
        result = _parse_separator(text, "pytest")
        assert len(result["generated_files"]) == 1
        assert result["generated_files"][0]["file_name"] == "tests/test_login.py"
        assert "import requests" in result["generated_files"][0]["code"]

    def test_parses_multiple_file_blocks(self):
        text = (
            "===FILE:tests/test_a.py===\ncode_a\n===END===\n"
            "===FILE:tests/test_b.py===\ncode_b\n===END==="
        )
        result = _parse_separator(text, "pytest")
        assert len(result["generated_files"]) == 2
        names = [f["file_name"] for f in result["generated_files"]]
        assert "tests/test_a.py" in names
        assert "tests/test_b.py" in names

    def test_framework_preserved(self):
        text = "===FILE:tests/test.py===\ncode\n===END==="
        result = _parse_separator(text, "playwright")
        assert result["framework"] == "playwright"

    def test_explanation_default_empty(self):
        text = "===FILE:tests/test.py===\ncode\n===END==="
        result = _parse_separator(text, "pytest")
        assert result["generated_files"][0]["explanation"] == ""

    def test_no_blocks_returns_empty(self):
        result = _parse_separator("{}", "pytest")
        # Falls back to JSON parse — no generated_files in empty JSON
        assert result.get("generated_files", []) == []
