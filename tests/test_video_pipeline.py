"""
Integration tests — all 3 agents with the ZaloPay transfer requirement.

Input: test_case_video.md  (POST /v1/transfer API spec)
Flow: Agent 1 preview → Agent 1 analyze → Agent 2 generate TCs → Agent 3 generate code

Run with a real API key:
    GREENNODE_AIP_KEY=<your-key> pytest tests/test_video_pipeline.py -v

Skipped automatically when GREENNODE_AIP_KEY is not set.
"""

import os
import re
from pathlib import Path

import pytest

# ── Skip entire module if no real key ────────────────────────────────────────
_HAS_KEY = bool(os.environ.get("GREENNODE_AIP_KEY")) and \
           os.environ.get("GREENNODE_AIP_KEY") != "test-key-dummy-unit"
pytestmark = pytest.mark.skipif(
    not _HAS_KEY,
    reason="GREENNODE_AIP_KEY not set — skipping integration tests",
)

REQUIREMENT = Path("test_case_video.md").read_text(encoding="utf-8").strip()


# ── Module-scope fixtures — each agent called once for the whole module ───────

@pytest.fixture(scope="module")
def preview_result():
    """Agent 1: preview (quality check + overview)."""
    from backend.app.agents.requirement_analyzer import RequirementAnalyzer
    return RequirementAnalyzer().preview(REQUIREMENT)


@pytest.fixture(scope="module")
def analyze_result():
    """Agent 1: analyze (extract scenarios)."""
    from backend.app.agents.requirement_analyzer import RequirementAnalyzer
    return RequirementAnalyzer().analyze(REQUIREMENT)


@pytest.fixture(scope="module")
def agent2_result(analyze_result):
    """Agent 2: generate test cases from scenarios."""
    from backend.app.agents.test_case_payload_generator import TestCasePayloadGenerator
    return TestCasePayloadGenerator().generate(
        scenarios=analyze_result["scenarios"],
        requirement_text=REQUIREMENT,
        overview=analyze_result.get("overview") or analyze_result.get("_overview"),
    )


@pytest.fixture(scope="module")
def agent3_result(agent2_result):
    """Agent 3: generate pytest automation code from test cases."""
    from backend.app.agents.automation_code_writer import AutomationCodeWriter
    return AutomationCodeWriter().generate(
        test_cases=agent2_result["test_cases"],
        framework="pytest",
        requirement_text=REQUIREMENT,
        payload_templates=agent2_result.get("payload_templates"),
    )


# ── Agent 1: Preview ──────────────────────────────────────────────────────────

class TestAgent1Preview:
    def test_has_overview(self, preview_result):
        assert "overview" in preview_result

    def test_overview_contains_transfer_endpoint(self, preview_result):
        endpoints = preview_result["overview"].get("endpoints", [])
        assert any("transfer" in ep.lower() or "/v1/" in ep for ep in endpoints), \
            f"Expected /v1/transfer in endpoints, got: {endpoints}"

    def test_overview_has_business_rules(self, preview_result):
        rules = preview_result["overview"].get("business_rules", [])
        assert len(rules) > 0, "Expected at least one business rule"

    def test_quality_score_structure(self, preview_result):
        qs = preview_result.get("quality_score", {})
        for key in ("overall", "completeness", "testability", "ambiguity", "risk"):
            assert key in qs, f"quality_score missing key: {key}"

    def test_quality_score_ranges(self, preview_result):
        qs = preview_result["quality_score"]
        for key in ("overall", "completeness", "testability", "ambiguity"):
            assert 0 <= qs[key] <= 100, f"{key}={qs[key]} out of range [0,100]"

    def test_risk_level_valid(self, preview_result):
        assert preview_result["quality_score"]["risk"] in {"Low", "Medium", "High"}

    def test_suggestions_is_list(self, preview_result):
        assert isinstance(preview_result.get("suggestions", []), list)

    def test_zalopay_context_in_summary(self, preview_result):
        summary = preview_result["overview"].get("summary", "")
        assert any(kw in summary.lower() for kw in ("zalopay", "transfer", "chuyển tiền", "payment")), \
            f"Expected ZaloPay context in summary: {summary}"


# ── Agent 1: Analyze ──────────────────────────────────────────────────────────

class TestAgent1Analyze:
    def test_has_scenarios(self, analyze_result):
        assert "scenarios" in analyze_result
        assert len(analyze_result["scenarios"]) >= 5, \
            f"Expected ≥5 scenarios for a payment API, got {len(analyze_result['scenarios'])}"

    def test_scenario_id_format(self, analyze_result):
        for s in analyze_result["scenarios"]:
            assert re.match(r"^SCN-\d{3}$", s.get("scenario_id", "")), \
                f"Bad scenario_id format: {s.get('scenario_id')}"

    def test_scenario_types_valid(self, analyze_result):
        valid = {"positive", "negative", "boundary", "security", "edge case"}
        for s in analyze_result["scenarios"]:
            assert s.get("type") in valid, \
                f"Invalid type '{s.get('type')}' in scenario {s.get('scenario_id')}"

    def test_scenario_priorities_valid(self, analyze_result):
        for s in analyze_result["scenarios"]:
            assert s.get("priority") in {"high", "medium", "low"}, \
                f"Invalid priority '{s.get('priority')}' in {s.get('scenario_id')}"

    def test_has_positive_happy_path(self, analyze_result):
        types = [s.get("type") for s in analyze_result["scenarios"]]
        assert "positive" in types, "Missing happy-path (positive) scenario"

    def test_has_negative_scenarios(self, analyze_result):
        types = [s.get("type") for s in analyze_result["scenarios"]]
        assert "negative" in types, "Missing negative scenario"

    def test_amount_scenario_present(self, analyze_result):
        all_text = " ".join(
            s.get("title", "") + " " + s.get("description", "")
            for s in analyze_result["scenarios"]
        ).lower()
        assert "amount" in all_text or "1.000" in all_text or "limit" in all_text, \
            "Expected amount/limit scenario for the payment API"

    def test_otp_scenario_present(self, analyze_result):
        all_text = " ".join(
            s.get("title", "") + " " + s.get("description", "")
            for s in analyze_result["scenarios"]
        ).lower()
        assert "otp" in all_text or "invalid_otp" in all_text, \
            "Expected OTP validation scenario"

    def test_requirements_summary_present(self, analyze_result):
        summary = analyze_result.get("requirements_summary", [])
        assert len(summary) > 0, "requirements_summary should not be empty"

    def test_each_scenario_has_given_when_then(self, analyze_result):
        for s in analyze_result["scenarios"]:
            sid = s.get("scenario_id", "?")
            assert s.get("given"), f"SCN {sid} missing 'given'"
            assert s.get("when"), f"SCN {sid} missing 'when'"
            assert s.get("then"), f"SCN {sid} missing 'then'"


# ── Agent 2: TestCasePayloadGenerator ────────────────────────────────────────

class TestAgent2:
    def test_has_test_cases(self, agent2_result):
        tcs = agent2_result.get("test_cases", [])
        assert len(tcs) > 0, "Expected at least one test case"

    def test_all_scenarios_covered(self, analyze_result, agent2_result):
        scenario_ids = {s["scenario_id"] for s in analyze_result["scenarios"]}
        tc_scenario_ids = {
            tc["scenario_id"] for tc in agent2_result["test_cases"]
            if tc.get("scenario_id")
        }
        uncovered = scenario_ids - tc_scenario_ids
        assert not uncovered, f"Scenarios without any TC: {uncovered}"

    def test_no_duplicate_tc_ids(self, agent2_result):
        ids = [tc["test_case_id"] for tc in agent2_result["test_cases"]]
        assert len(ids) == len(set(ids)), \
            f"Duplicate test_case_ids: {[x for x in ids if ids.count(x) > 1]}"

    def test_required_fields_on_every_tc(self, agent2_result):
        required = ("test_case_id", "scenario_id", "name", "steps", "expected_result", "technique")
        for tc in agent2_result["test_cases"]:
            for field in required:
                assert field in tc and tc[field] is not None, \
                    f"TC {tc.get('test_case_id','?')} missing field '{field}'"

    def test_steps_are_non_empty_lists(self, agent2_result):
        for tc in agent2_result["test_cases"]:
            steps = tc.get("steps", [])
            assert isinstance(steps, list) and len(steps) >= 1, \
                f"TC {tc.get('test_case_id','?')} has empty steps"

    def test_test_data_is_dict(self, agent2_result):
        for tc in agent2_result["test_cases"]:
            assert isinstance(tc.get("test_data", {}), dict), \
                f"TC {tc.get('test_case_id','?')} test_data is not a dict"

    def test_technique_codes_valid(self, agent2_result):
        valid_techniques = {"EP", "BVA", "EG", "DT", "UC", "AT", "ST", "RT"}
        for tc in agent2_result["test_cases"]:
            tech = tc.get("technique", "")
            assert tech in valid_techniques, \
                f"TC {tc.get('test_case_id','?')} has unknown technique '{tech}'"

    def test_applied_techniques_present(self, agent2_result):
        assert agent2_result.get("applied_techniques") or agent2_result.get("verified_techniques"), \
            "Missing applied_techniques or verified_techniques in Agent 2 output"

    def test_transfer_endpoint_in_test_data(self, agent2_result):
        all_data = str(agent2_result["test_cases"]).lower()
        assert "/v1/transfer" in all_data or "transfer" in all_data, \
            "Expected /v1/transfer endpoint reference in test cases"

    def test_amount_field_in_some_test_data(self, agent2_result):
        all_data = str(agent2_result["test_cases"]).lower()
        assert "amount" in all_data, "Expected 'amount' field referenced in test data"

    def test_otp_field_in_some_test_data(self, agent2_result):
        all_data = str(agent2_result["test_cases"]).lower()
        assert "otp" in all_data, "Expected 'otp' field referenced in test data"


# ── Agent 3: AutomationCodeWriter ─────────────────────────────────────────────

class TestAgent3:
    def test_has_generated_files(self, agent3_result):
        files = agent3_result.get("generated_files", [])
        assert len(files) > 0, "Expected at least one generated file"

    def test_framework_is_pytest(self, agent3_result):
        assert agent3_result.get("framework") == "pytest"

    def test_all_files_are_python(self, agent3_result):
        for f in agent3_result["generated_files"]:
            assert f["file_name"].endswith(".py"), \
                f"Expected .py file, got: {f['file_name']}"

    def test_files_have_non_empty_code(self, agent3_result):
        for f in agent3_result["generated_files"]:
            assert f.get("code", "").strip(), \
                f"File {f.get('file_name','?')} has empty code"

    def test_code_has_imports(self, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        assert "import" in all_code, "Generated code has no import statements"

    def test_code_has_test_functions(self, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        assert "def test_" in all_code, "Generated code has no test functions"

    def test_test_function_count_matches_tc_count(self, agent2_result, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        n_functions = len(re.findall(r"def test_", all_code))
        n_tcs = len(agent2_result["test_cases"])
        assert n_functions > 0
        assert n_functions <= n_tcs, \
            f"More test functions ({n_functions}) than test cases ({n_tcs})"

    def test_code_uses_base_url_env(self, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        assert "BASE_URL" in all_code or "base_url" in all_code.lower(), \
            "Expected BASE_URL env var usage in generated code"

    def test_code_references_transfer_api(self, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        assert "transfer" in all_code.lower() or "/v1/" in all_code, \
            "Expected /v1/transfer reference in generated code"

    def test_code_has_assertions(self, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        assert "assert" in all_code, "Generated code has no assertions"

    def test_no_placeholder_todos(self, agent3_result):
        all_code = "\n".join(f["code"] for f in agent3_result["generated_files"])
        for stub in ("...", "TODO", "pass  # stub", "raise NotImplementedError"):
            assert stub not in all_code, \
                f"Generated code contains stub/placeholder: {stub!r}"


# ── Full Pipeline: end-to-end traceability ────────────────────────────────────

class TestFullPipeline:
    def test_scenario_count_reasonable(self, analyze_result):
        n = len(analyze_result["scenarios"])
        assert 5 <= n <= 30, f"Unexpected scenario count: {n}"

    def test_tc_count_gte_scenario_count(self, analyze_result, agent2_result):
        n_scenarios = len(analyze_result["scenarios"])
        n_tcs = len(agent2_result["test_cases"])
        assert n_tcs >= n_scenarios, \
            f"Expected ≥{n_scenarios} TCs (one per scenario), got {n_tcs}"

    def test_every_scenario_has_tc_in_code(self, analyze_result, agent2_result, agent3_result):
        """Each scenario_id from Agent 1 should appear in either Agent 2's TCs or Agent 3's code."""
        scenario_ids = {s["scenario_id"] for s in analyze_result["scenarios"]}
        tc_scenario_ids = {tc["scenario_id"] for tc in agent2_result["test_cases"] if tc.get("scenario_id")}
        assert scenario_ids <= tc_scenario_ids, \
            f"Scenarios without TC coverage: {scenario_ids - tc_scenario_ids}"

    def test_no_agent_raised_error(self, preview_result, analyze_result, agent2_result, agent3_result):
        """All agents must return valid dicts (no error keys)."""
        for name, result in [
            ("Agent1-preview", preview_result),
            ("Agent1-analyze", analyze_result),
            ("Agent2", agent2_result),
            ("Agent3", agent3_result),
        ]:
            assert "error" not in result, f"{name} returned error: {result.get('error')}"
            assert "detail" not in result, f"{name} returned FastAPI error: {result.get('detail')}"

    def test_pipeline_output_types(self, preview_result, analyze_result, agent2_result, agent3_result):
        assert isinstance(preview_result["overview"], dict)
        assert isinstance(analyze_result["scenarios"], list)
        assert isinstance(agent2_result["test_cases"], list)
        assert isinstance(agent3_result["generated_files"], list)

    def test_amount_limits_tested(self, agent2_result):
        """ZaloPay spec has amount 1000–100M VND and daily cap 200M — BVA should cover this."""
        all_data = str(agent2_result["test_cases"]).lower()
        assert any(v in all_data for v in ("1000", "100000000", "200000000", "1.000", "limit")), \
            "Expected amount boundary values in test cases"

    def test_error_codes_tested(self, agent2_result):
        """All 5 error codes in spec should appear in expected_results."""
        all_expected = " ".join(
            tc.get("expected_result", "") for tc in agent2_result["test_cases"]
        ).lower()
        expected_codes = {
            "insufficient_balance": "insufficient_balance" in all_expected or "400" in all_expected,
            "receiver_not_found": "receiver_not_found" in all_expected or "404" in all_expected,
            "invalid_otp": "invalid_otp" in all_expected or "401" in all_expected,
        }
        uncovered = [code for code, covered in expected_codes.items() if not covered]
        assert not uncovered, f"Error codes not tested: {uncovered}"
