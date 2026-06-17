import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

_logger = logging.getLogger(__name__)


class RequirementInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=8000, description="Raw requirement or spec text")
    source: Optional[str] = Field(default="manual", description="Source of the input")
    framework: Optional[str] = Field(default="pytest", description="Test framework for code generation")


class RequirementItem(BaseModel):
    id: str = ""
    text: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v):
        if v is None:
            return ""
        s = str(v).strip()
        if not s:
            _logger.warning("RequirementItem.id is empty — LLM may have used a different field name")
        return s


class OverviewItem(BaseModel):
    summary: str = ""
    features: List[str] = []
    endpoints: List[str] = []
    business_rules: List[str] = []


class SuggestionItem(BaseModel):
    type: str = "improvement"
    title: str = ""
    description: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        if not isinstance(v, str):
            _logger.warning("SuggestionItem.type received non-string %r — defaulting to 'improvement'", v)
            return "improvement"
        normalized = v.lower().strip()
        if normalized not in {"missing", "improvement", "ambiguity"}:
            _logger.warning("SuggestionItem.type unknown value %r — defaulting to 'improvement'", v)
            return "improvement"
        return normalized


class ScoreBreakdown(BaseModel):
    completeness_found: List[str] = []
    completeness_missing: List[str] = []
    testability_found: List[str] = []
    testability_missing: List[str] = []
    clarity_found: List[str] = []
    clarity_missing: List[str] = []


class QualityScore(BaseModel):
    overall: int = 0
    completeness: int = 0
    testability: int = 0
    ambiguity: int = 0
    risk: str = "Medium"
    score_breakdown: ScoreBreakdown = ScoreBreakdown()

    @field_validator("overall", "completeness", "testability", "ambiguity", mode="before")
    @classmethod
    def clamp_score(cls, v):
        try:
            return max(0, min(100, int(v)))
        except Exception:
            return 0

    @field_validator("score_breakdown", mode="before")
    @classmethod
    def coerce_breakdown(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("risk", mode="before")
    @classmethod
    def normalize_risk(cls, v):
        if not isinstance(v, str):
            return "Medium"
        v = v.strip().capitalize()
        return v if v in {"Low", "Medium", "High"} else "Medium"


class RequirementPreviewResult(BaseModel):
    overview: OverviewItem
    suggestions: List[SuggestionItem] = []
    quality_score: Optional[QualityScore] = None


class ScenarioItem(BaseModel):
    scenario_id: str
    title: str
    description: str = ""
    given: str = ""
    when: str = ""
    then: str = ""
    priority: str = "medium"
    type: str = "positive"
    related_requirement: Optional[str] = None
    related_endpoint: Optional[str] = None

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, v):
        if not isinstance(v, str):
            return "medium"
        v = v.lower().strip()
        return v if v in {"high", "medium", "low"} else "medium"

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        if not isinstance(v, str):
            return "positive"
        v = v.lower().strip().replace("_", " ")
        return v if v in {"positive", "negative", "boundary", "security", "edge case"} else "positive"


class RequirementAnalysisResult(BaseModel):
    requirements_summary: List[RequirementItem] = []
    scenarios: List[ScenarioItem]
    missing_information: List[str] = []


class AppliedTechnique(BaseModel):
    technique: str
    rationale: str = ""
    applicable_scenarios: List[str] = []

    @field_validator("technique", "rationale", mode="before")
    @classmethod
    def coerce_str(cls, v):
        return str(v) if v is not None else ""

    @field_validator("applicable_scenarios", mode="before")
    @classmethod
    def coerce_scenarios(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []


class TestCaseItem(BaseModel):
    test_case_id: str
    name: str
    preconditions: str = ""
    steps: List[str] = []
    test_data: Dict[str, Any] = {}
    expected_result: str = ""
    priority: str = "medium"
    tags: List[str] = []
    technique: Optional[str] = None
    scenario_id: Optional[str] = None

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, v):
        if not isinstance(v, str):
            return "medium"
        v = v.lower().strip()
        return v if v in {"high", "medium", "low"} else "medium"

    @field_validator("test_data", mode="before")
    @classmethod
    def coerce_test_data(cls, v):
        if isinstance(v, str):
            try:
                import json
                parsed = json.loads(v)
                return parsed if isinstance(parsed, dict) else {"value": v}
            except Exception:
                return {"value": v}
        if v is None:
            return {}
        if isinstance(v, list):
            # A bare list has no field names — discard rather than produce {"0": ...} noise
            return {}
        if not isinstance(v, dict):
            return {"value": str(v)}
        return v

    @field_validator("steps", mode="before")
    @classmethod
    def coerce_steps(cls, v):
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v):
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v


class TestCasePayloadResult(BaseModel):
    system_type: str = "traditional"
    applied_techniques: List[AppliedTechnique] = []
    test_cases: List[TestCaseItem] = []
    test_data_matrix: List[Dict[str, Any]] = []
    payload_templates: List[Dict[str, Any]] = []


class GenerateTestCasesInput(BaseModel):
    scenarios: List[ScenarioItem]
    requirement_text: Optional[str] = None
    overview: Optional[dict] = None


class GenerateAutomationCodeInput(BaseModel):
    test_cases: List[TestCaseItem]
    framework: str = "pytest"
    requirement_text: Optional[str] = None
    payload_templates: Optional[List[Dict[str, Any]]] = None
    test_data_matrix: Optional[List[Dict[str, Any]]] = None


class GeneratedCodeFile(BaseModel):
    file_name: str
    code: str
    explanation: str = ""


class AutomationCodeResult(BaseModel):
    framework: str
    generated_files: List[GeneratedCodeFile]
