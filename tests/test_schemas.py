"""Tests for Pydantic schema validation and field coercion."""
import pytest
from backend.app.schemas.agent_schemas import (
    QualityScore,
    ScoreBreakdown,
    ScenarioItem,
    TestCaseItem,
    AppliedTechnique,
    RequirementAnalysisResult,
    TestCasePayloadResult,
    SuggestionItem,
)


# ── QualityScore ─────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_scores_clamped_to_100(self):
        qs = QualityScore(overall=150, completeness=200, testability=999, ambiguity=101)
        assert qs.overall == 100
        assert qs.completeness == 100
        assert qs.testability == 100
        assert qs.ambiguity == 100

    def test_scores_clamped_to_0(self):
        qs = QualityScore(overall=-10, completeness=-1)
        assert qs.overall == 0
        assert qs.completeness == 0

    def test_risk_normalization(self):
        assert QualityScore(risk="low").risk == "Low"
        assert QualityScore(risk="HIGH").risk == "High"
        assert QualityScore(risk="medium").risk == "Medium"
        assert QualityScore(risk="unknown").risk == "Medium"
        assert QualityScore(risk=None).risk == "Medium"

    def test_score_breakdown_default(self):
        qs = QualityScore()
        assert isinstance(qs.score_breakdown, ScoreBreakdown)
        assert qs.score_breakdown.completeness_found == []

    def test_score_breakdown_from_dict(self):
        bd = {"completeness_found": ["a", "b"], "testability_missing": ["c"]}
        qs = QualityScore(score_breakdown=bd)
        assert qs.score_breakdown.completeness_found == ["a", "b"]
        assert qs.score_breakdown.testability_missing == ["c"]

    def test_score_breakdown_from_none(self):
        qs = QualityScore(score_breakdown=None)
        assert qs.score_breakdown.completeness_found == []

    def test_nonnumeric_score_defaults_to_0(self):
        qs = QualityScore(overall="bad", completeness="N/A")
        assert qs.overall == 0
        assert qs.completeness == 0


# ── ScenarioItem ─────────────────────────────────────────────────────────────

class TestScenarioItem:
    def _make(self, **kw):
        defaults = dict(scenario_id="SCN-001", title="Test scenario")
        return ScenarioItem(**{**defaults, **kw})

    def test_priority_normalization(self):
        assert self._make(priority="HIGH").priority == "high"
        assert self._make(priority="Medium").priority == "medium"
        assert self._make(priority="LOW").priority == "low"
        assert self._make(priority="critical").priority == "medium"
        assert self._make(priority=None).priority == "medium"

    def test_type_normalization(self):
        assert self._make(type="POSITIVE").type == "positive"
        assert self._make(type="Negative").type == "negative"
        assert self._make(type="edge_case").type == "edge case"
        assert self._make(type="boundary").type == "boundary"
        assert self._make(type="security").type == "security"
        assert self._make(type="invalid").type == "positive"

    def test_defaults(self):
        s = self._make()
        assert s.given == ""
        assert s.when == ""
        assert s.then == ""
        assert s.related_requirement is None
        assert s.related_endpoint is None


# ── TestCaseItem ─────────────────────────────────────────────────────────────

class TestTestCaseItem:
    def _make(self, **kw):
        defaults = dict(test_case_id="TC-EP-001", name="valid login")
        return TestCaseItem(**{**defaults, **kw})

    def test_test_data_from_string_json(self):
        tc = self._make(test_data='{"email": "a@b.com"}')
        assert tc.test_data == {"email": "a@b.com"}

    def test_test_data_from_plain_string(self):
        tc = self._make(test_data="some value")
        assert tc.test_data == {"value": "some value"}

    def test_test_data_from_list(self):
        # List has no field names — coerced to empty dict to avoid meaningless {"0": ...} keys
        tc = self._make(test_data=["a", "b"])
        assert tc.test_data == {}

    def test_test_data_from_none(self):
        tc = self._make(test_data=None)
        assert tc.test_data == {}

    def test_steps_from_string(self):
        tc = self._make(steps="do the thing")
        assert tc.steps == ["do the thing"]

    def test_steps_from_none(self):
        tc = self._make(steps=None)
        assert tc.steps == []

    def test_tags_from_string(self):
        tc = self._make(tags="positive")
        assert tc.tags == ["positive"]

    def test_tags_from_none(self):
        tc = self._make(tags=None)
        assert tc.tags == []

    def test_priority_normalization(self):
        assert self._make(priority="HIGH").priority == "high"
        assert self._make(priority="critical").priority == "medium"

    def test_optional_fields_default_none(self):
        tc = self._make()
        assert tc.technique is None
        assert tc.scenario_id is None


# ── AppliedTechnique ──────────────────────────────────────────────────────────

class TestAppliedTechnique:
    def test_applicable_scenarios_from_comma_string(self):
        t = AppliedTechnique(technique="EP", applicable_scenarios="SCN-001, SCN-002, SCN-003")
        assert t.applicable_scenarios == ["SCN-001", "SCN-002", "SCN-003"]

    def test_applicable_scenarios_from_list(self):
        t = AppliedTechnique(technique="BVA", applicable_scenarios=["SCN-001", "SCN-002"])
        assert t.applicable_scenarios == ["SCN-001", "SCN-002"]

    def test_applicable_scenarios_from_none(self):
        t = AppliedTechnique(technique="DT", applicable_scenarios=None)
        assert t.applicable_scenarios == []

    def test_technique_coerced_to_str(self):
        t = AppliedTechnique(technique=None)
        assert t.technique == ""


# ── RequirementAnalysisResult ─────────────────────────────────────────────────

class TestRequirementAnalysisResult:
    def test_defaults_for_optional_lists(self):
        r = RequirementAnalysisResult(scenarios=[])
        assert r.requirements_summary == []
        assert r.missing_information == []

    def test_scenarios_required(self):
        with pytest.raises(Exception):
            RequirementAnalysisResult()


# ── SuggestionItem ────────────────────────────────────────────────────────────

class TestSuggestionItem:
    def test_type_normalization(self):
        assert SuggestionItem(type="missing").type == "missing"
        assert SuggestionItem(type="AMBIGUITY").type == "ambiguity"
        assert SuggestionItem(type="unknown").type == "improvement"
        assert SuggestionItem(type=None).type == "improvement"
