"""Tests for workflow logic — LLM calls are mocked."""
import pytest
from unittest.mock import MagicMock, patch

from backend.app.schemas.agent_schemas import ScenarioItem, TestCaseItem, TestCasePayloadResult
from backend.app.workflows.mvp_workflow import MVPWorkflow
from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow


# ── MVPWorkflow ───────────────────────────────────────────────────────────────

class TestMVPWorkflow:
    def _make_workflow(self, analyze_return):
        wf = MVPWorkflow.__new__(MVPWorkflow)
        wf.analyzer = MagicMock()
        wf.analyzer.analyze.return_value = analyze_return
        return wf

    def test_happy_path(self):
        wf = self._make_workflow({
            "scenarios": [
                {"scenario_id": "SCN-001", "title": "Valid login",
                 "priority": "high", "type": "positive"}
            ],
            "requirements_summary": [{"id": "REQ-001", "text": "Login"}],
            "missing_information": [],
        })
        result = wf.run("Login requirement")
        assert len(result.scenarios) == 1
        assert result.scenarios[0].scenario_id == "SCN-001"

    def test_missing_scenarios_key_defaults_to_empty(self):
        wf = self._make_workflow({"requirements_summary": []})
        result = wf.run("Some req")
        assert result.scenarios == []

    def test_malformed_response_returns_fallback(self):
        wf = self._make_workflow({"scenarios": "not-a-list"})
        result = wf.run("Some req")
        # Pydantic raises, fallback kicks in
        assert result.scenarios == []
        assert "unexpected response format" in result.missing_information[0].lower()

    def test_missing_information_defaults(self):
        wf = self._make_workflow({"scenarios": []})
        result = wf.run("req")
        assert result.missing_information == []

    def test_requirements_summary_defaults(self):
        wf = self._make_workflow({"scenarios": []})
        result = wf.run("req")
        assert result.requirements_summary == []


# ── TestCasePayloadWorkflow grounding ─────────────────────────────────────────

class TestGroundingValidation:
    def _make_tc(self, tc_id, scenario_id, technique="EP"):
        return TestCaseItem(
            test_case_id=tc_id,
            name="test",
            scenario_id=scenario_id,
            technique=technique,
        )

    def _scenarios(self, ids):
        return [
            ScenarioItem(scenario_id=sid, title=f"Scenario {sid}")
            for sid in ids
        ]

    def test_drops_hallucinated_scenario_ids(self):
        from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)

        from backend.app.schemas.agent_schemas import TestCasePayloadResult
        result = TestCasePayloadResult(
            test_cases=[
                self._make_tc("TC-EP-001", "SCN-001"),
                self._make_tc("TC-EP-002", "SCN-999"),  # hallucinated
            ],
        )
        scenarios = self._scenarios(["SCN-001"])

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wf._validate_grounding(scenarios, result)
            assert any("dropped" in str(warning.message).lower() for warning in w)

        assert len(result.test_cases) == 1
        assert result.test_cases[0].scenario_id == "SCN-001"

    def test_adds_fallback_for_uncovered_scenario(self):
        from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)

        from backend.app.schemas.agent_schemas import TestCasePayloadResult
        result = TestCasePayloadResult(
            test_cases=[self._make_tc("TC-EP-001", "SCN-001")],
        )
        scenarios = self._scenarios(["SCN-001", "SCN-002"])

        wf._validate_grounding(scenarios, result)

        assert len(result.test_cases) == 2
        covered = {tc.scenario_id for tc in result.test_cases}
        assert "SCN-002" in covered
        # Fallback must be a proper TestCaseItem, not a raw dict
        fallback = next(tc for tc in result.test_cases if tc.scenario_id == "SCN-002")
        assert isinstance(fallback, TestCaseItem)

    def test_no_changes_when_all_covered(self):
        from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)

        from backend.app.schemas.agent_schemas import TestCasePayloadResult
        result = TestCasePayloadResult(
            test_cases=[
                self._make_tc("TC-EP-001", "SCN-001"),
                self._make_tc("TC-EP-002", "SCN-002"),
            ],
        )
        scenarios = self._scenarios(["SCN-001", "SCN-002"])

        wf._validate_grounding(scenarios, result)
        assert len(result.test_cases) == 2

    def test_tc_with_none_scenario_id_is_dropped(self):
        # TC with no scenario_id is treated as hallucinated and removed;
        # the covered scenario still has its valid TC so no fallback is added.
        from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)

        from backend.app.schemas.agent_schemas import TestCasePayloadResult
        result = TestCasePayloadResult(
            test_cases=[
                self._make_tc("TC-EP-001", "SCN-001"),
                self._make_tc("TC-UC-001", None),  # no scenario_id — should be dropped
            ],
        )
        scenarios = self._scenarios(["SCN-001"])
        wf._validate_grounding(scenarios, result)
        # None-scenario_id TC is dropped; SCN-001 is still covered so no fallback added
        assert len(result.test_cases) == 1
        assert result.test_cases[0].test_case_id == "TC-EP-001"


# ── Deduplication — string fallback (no generator) ────────────────────────────

class TestDeduplicationStringFallback:
    """Tests run via _deduplicate_test_cases_string directly (no LLM needed)."""

    def _wf(self):
        return TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)

    def _tc(self, tc_id, sid, name, expected="200 OK", steps=None, test_data=None, priority="medium", technique="EP"):
        return TestCaseItem(
            test_case_id=tc_id, name=name, scenario_id=sid,
            technique=technique, expected_result=expected,
            steps=steps or [], test_data=test_data or {},
            priority=priority,
        )

    def _result(self, tcs):
        return TestCasePayloadResult(test_cases=tcs)

    def test_no_duplicates_unchanged(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-EP-002", "SCN-001", "Invalid password"),
            self._tc("TC-EP-003", "SCN-002", "Valid registration"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 3

    def test_exact_duplicate_removed(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "Valid login", "200 OK"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 1

    def test_keeps_richer_duplicate(self):
        wf = self._wf()
        sparse = self._tc("TC-EP-001", "SCN-001", "Valid login", "200 OK",
                          steps=["step1"], test_data={}, priority="low")
        rich = self._tc("TC-BVA-001", "SCN-001", "Valid login", "200 OK",
                        steps=["step1", "step2", "step3"],
                        test_data={"email": "a@b.com"}, priority="high")
        r = self._result([sparse, rich])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 1
        assert r.test_cases[0].test_case_id == "TC-BVA-001"

    def test_case_insensitive_name_match(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid Login", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "valid login", "200 OK"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 1

    def test_punctuation_ignored_in_name(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid Login!", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "Valid Login.", "200 OK"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 1

    def test_different_scenarios_not_deduplicated(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid input", "200 OK"),
            self._tc("TC-EP-002", "SCN-002", "Valid input", "200 OK"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 2

    def test_different_expected_results_not_deduplicated(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Login attempt", "200 OK"),
            self._tc("TC-EP-002", "SCN-001", "Login attempt", "401 Unauthorized"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 2

    def test_multiple_duplicates_all_removed(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "Valid login", "200 OK"),
            self._tc("TC-DT-001", "SCN-001", "Valid login", "200 OK"),
        ])
        wf._deduplicate_test_cases_string(r)
        assert len(r.test_cases) == 1

    def test_empty_list_unchanged(self):
        wf = self._wf()
        r = self._result([])
        wf._deduplicate_test_cases(r)
        assert r.test_cases == []

    def test_string_dedup_warning_issued(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "Valid login", "200 OK"),
        ])
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            wf._deduplicate_test_cases_string(r)
            assert any("dedup" in str(x.message).lower() for x in w)

    def test_no_warning_when_no_duplicates(self):
        wf = self._wf()
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-EP-002", "SCN-001", "Invalid login"),
        ])
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            wf._deduplicate_test_cases_string(r)
            assert not any("dedup" in str(x.message).lower() for x in w)


# ── Deduplication — AI path (mocked LLM) ──────────────────────────────────────

class TestDeduplicationAI:
    def _wf_with_mock_generator(self, llm_response: dict):
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)
        wf.generator = MagicMock()
        wf.generator._call.return_value = llm_response
        return wf

    def _tc(self, tc_id, sid, name, expected="200 OK", steps=None, technique="EP"):
        return TestCaseItem(
            test_case_id=tc_id, name=name, scenario_id=sid,
            technique=technique, expected_result=expected,
            steps=steps or [], test_data={}, priority="medium",
        )

    def _result(self, tcs):
        return TestCasePayloadResult(test_cases=tcs)

    def test_ai_removes_duplicate(self):
        wf = self._wf_with_mock_generator({
            "keep": ["TC-EP-001"],
            "removed": [{"id": "TC-BVA-001", "reason": "same scenario and result"}],
        })
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-BVA-001", "SCN-001", "Valid login — boundary"),
        ])
        wf._deduplicate_test_cases_ai(r)
        assert len(r.test_cases) == 1
        assert r.test_cases[0].test_case_id == "TC-EP-001"

    def test_ai_no_duplicates_keeps_all(self):
        wf = self._wf_with_mock_generator({
            "keep": ["TC-EP-001", "TC-EP-002"],
            "removed": [],
        })
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-EP-002", "SCN-001", "Invalid login"),
        ])
        wf._deduplicate_test_cases_ai(r)
        assert len(r.test_cases) == 2

    def test_ai_warning_per_removed_tc(self):
        wf = self._wf_with_mock_generator({
            "keep": ["TC-EP-001"],
            "removed": [
                {"id": "TC-BVA-001", "reason": "dup A"},
                {"id": "TC-DT-001", "reason": "dup B"},
            ],
        })
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-BVA-001", "SCN-001", "Valid login v2"),
            self._tc("TC-DT-001", "SCN-001", "Valid login v3"),
        ])
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            wf._deduplicate_test_cases_ai(r)
            dedup_warns = [x for x in w if "dedup (ai)" in str(x.message).lower()]
            assert len(dedup_warns) == 2

    def test_ai_unknown_id_raises(self):
        wf = self._wf_with_mock_generator({
            "keep": ["TC-EP-001", "GHOST-999"],
            "removed": [],
        })
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-EP-002", "SCN-001", "Invalid login"),
        ])
        import pytest as _pytest
        with _pytest.raises(ValueError, match="unknown IDs"):
            wf._deduplicate_test_cases_ai(r)

    def test_ai_missing_id_raises(self):
        wf = self._wf_with_mock_generator({
            "keep": ["TC-EP-001"],
            "removed": [],  # TC-EP-002 not accounted for
        })
        r = self._result([
            self._tc("TC-EP-001", "SCN-001", "Valid login"),
            self._tc("TC-EP-002", "SCN-001", "Invalid login"),
        ])
        import pytest as _pytest
        with _pytest.raises(ValueError, match="not account"):
            wf._deduplicate_test_cases_ai(r)

    def test_dispatcher_falls_back_on_llm_error(self):
        """When AI call fails, dispatcher falls back to string dedup silently."""
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)
        wf.generator = MagicMock()
        wf.generator._call.side_effect = RuntimeError("rate limit")
        r = TestCasePayloadResult(test_cases=[
            self._tc("TC-EP-001", "SCN-001", "Valid login", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "valid login", "200 OK"),  # string dup
        ])
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            wf._deduplicate_test_cases(r)
        assert any("ai dedup unavailable" in str(x.message).lower() for x in w)
        assert len(r.test_cases) == 1  # string fallback removed the dup

    def test_dispatcher_falls_back_when_no_generator(self):
        """When generator attribute is missing (unit test env), falls back to string dedup."""
        wf = TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)  # no __init__
        r = TestCasePayloadResult(test_cases=[
            self._tc("TC-EP-001", "SCN-001", "Valid login", "200 OK"),
            self._tc("TC-BVA-001", "SCN-001", "Valid login", "200 OK"),
        ])
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            wf._deduplicate_test_cases(r)
        assert any("ai dedup unavailable" in str(x.message).lower() for x in w)
        assert len(r.test_cases) == 1


# ── TC ID rebuild ─────────────────────────────────────────────────────────────

class TestTCIdRebuild:
    """Verify the post-processing loop in TestCasePayloadGenerator."""

    def _rebuild(self, test_cases):
        from typing import Dict
        tech_counters: Dict[str, int] = {}
        for tc in test_cases:
            tech = (tc.get("technique") or "UC").upper()
            tech_counters[tech] = tech_counters.get(tech, 0) + 1
            tc["test_case_id"] = f"TC-{tech}-{tech_counters[tech]:03d}"
            tc["technique"] = tech
        return test_cases

    def test_ids_follow_tc_technique_nnn_format(self):
        tcs = [
            {"technique": "EP", "name": "a"},
            {"technique": "BVA", "name": "b"},
            {"technique": "EP", "name": "c"},
        ]
        result = self._rebuild(tcs)
        assert result[0]["test_case_id"] == "TC-EP-001"
        assert result[1]["test_case_id"] == "TC-BVA-001"
        assert result[2]["test_case_id"] == "TC-EP-002"

    def test_null_technique_defaults_to_uc(self):
        tcs = [{"technique": None, "name": "a"}]
        result = self._rebuild(tcs)
        assert result[0]["test_case_id"] == "TC-UC-001"
        assert result[0]["technique"] == "UC"

    def test_empty_technique_defaults_to_uc(self):
        tcs = [{"technique": "", "name": "a"}]
        result = self._rebuild(tcs)
        assert result[0]["technique"] == "UC"

    def test_technique_field_normalized_to_uppercase(self):
        tcs = [{"technique": "ep", "name": "a"}]
        result = self._rebuild(tcs)
        assert result[0]["technique"] == "EP"

    def test_counters_are_per_technique(self):
        tcs = [{"technique": "EG"}, {"technique": "EG"}, {"technique": "EG"}]
        result = self._rebuild(tcs)
        ids = [tc["test_case_id"] for tc in result]
        assert ids == ["TC-EG-001", "TC-EG-002", "TC-EG-003"]


# ── _build_payload_templates ──────────────────────────────────────────────────

class TestBuildPayloadTemplates:
    def _wf(self):
        return TestCasePayloadWorkflow.__new__(TestCasePayloadWorkflow)

    def _tc(self, tc_id, sid, test_data=None, technique="EP", priority="high", name="test"):
        return TestCaseItem(
            test_case_id=tc_id,
            name=name,
            scenario_id=sid,
            technique=technique,
            priority=priority,
            test_data=test_data or {},
        )

    def _result(self, test_cases, payload_templates=None):
        return TestCasePayloadResult(
            test_cases=test_cases,
            payload_templates=payload_templates or [],
        )

    def test_builds_from_test_data(self):
        wf = self._wf()
        result = self._result([
            self._tc("TC-EP-001", "SCN-001", {"email": "a@b.com", "password": "123"}),
        ])
        wf._build_payload_templates(result)
        assert len(result.payload_templates) == 1

    def test_template_has_required_fields(self):
        wf = self._wf()
        result = self._result([
            self._tc("TC-EP-001", "SCN-001", {"amount": 100}, technique="EP", priority="high", name="Pay"),
        ])
        wf._build_payload_templates(result)
        tpl = result.payload_templates[0]
        assert tpl["test_case_id"] == "TC-EP-001"
        assert tpl["scenario_id"] == "SCN-001"
        assert tpl["name"] == "Pay"
        assert tpl["payload"] == {"amount": 100}
        assert tpl["technique"] == "EP"
        assert tpl["priority"] == "high"

    def test_skips_tc_with_empty_test_data(self):
        wf = self._wf()
        result = self._result([
            self._tc("TC-EP-001", "SCN-001", {}),
            self._tc("TC-EP-002", "SCN-002", {"key": "val"}),
        ])
        wf._build_payload_templates(result)
        assert len(result.payload_templates) == 1
        assert result.payload_templates[0]["test_case_id"] == "TC-EP-002"

    def test_skips_if_already_populated(self):
        wf = self._wf()
        existing = [{"test_case_id": "TC-EP-001", "payload": {"x": 1}}]
        result = self._result(
            [self._tc("TC-EP-001", "SCN-001", {"email": "new@b.com"})],
            payload_templates=existing,
        )
        wf._build_payload_templates(result)
        # Should not overwrite
        assert result.payload_templates == existing

    def test_empty_test_cases_gives_empty_templates(self):
        wf = self._wf()
        result = self._result([])
        wf._build_payload_templates(result)
        assert result.payload_templates == []

    def test_multiple_tcs_all_with_data(self):
        wf = self._wf()
        result = self._result([
            self._tc("TC-EP-001", "SCN-001", {"a": 1}),
            self._tc("TC-BVA-001", "SCN-002", {"b": 2}),
            self._tc("TC-EG-001", "SCN-003", {"c": 3}),
        ])
        wf._build_payload_templates(result)
        assert len(result.payload_templates) == 3
        ids = [t["test_case_id"] for t in result.payload_templates]
        assert "TC-EP-001" in ids
        assert "TC-BVA-001" in ids
        assert "TC-EG-001" in ids
