"""Tests for FastAPI endpoints — all LLM/workflow calls are mocked."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    mock_workflow = MagicMock()
    mock_tc_workflow = MagicMock()
    mock_code_workflow = MagicMock()
    mock_full_workflow = MagicMock()

    from backend.app.schemas.agent_schemas import (
        RequirementAnalysisResult, TestCasePayloadResult, AutomationCodeResult,
        ScenarioItem, TestCaseItem, AppliedTechnique, GeneratedCodeFile,
    )

    # Agent 1 analyze mock
    mock_workflow.run.return_value = RequirementAnalysisResult(
        scenarios=[ScenarioItem(scenario_id="SCN-001", title="Valid login")],
        requirements_summary=[],
        missing_information=[],
    )
    # Agent 1 preview mock
    mock_workflow.analyzer = MagicMock()
    mock_workflow.analyzer.preview.return_value = {
        "overview": {"summary": "Login API", "features": [], "endpoints": [], "business_rules": []},
        "suggestions": [],
        "quality_score": {"overall": 80, "completeness": 80, "testability": 75,
                          "ambiguity": 85, "risk": "Low",
                          "score_breakdown": {"completeness_found": [], "completeness_missing": [],
                                              "testability_found": [], "testability_missing": [],
                                              "ambiguity_deductions": [], "clarity_found": [],
                                              "clarity_missing": []}},
        "quality_checks": {},
    }

    # Agent 2 mock
    mock_tc_workflow.run.return_value = TestCasePayloadResult(
        system_type="traditional",
        applied_techniques=[AppliedTechnique(technique="EP", rationale="partitions")],
        test_cases=[TestCaseItem(test_case_id="TC-EP-001", name="Valid login",
                                 scenario_id="SCN-001", technique="EP")],
    )

    # Agent 3 mock
    mock_code_workflow.run.return_value = AutomationCodeResult(
        framework="pytest",
        generated_files=[GeneratedCodeFile(file_name="tests/test_login.py",
                                           code="def test_login(): pass", explanation="")],
    )

    with patch.multiple(
        "backend.main",
        workflow=mock_workflow,
        test_case_workflow=mock_tc_workflow,
        automation_code_workflow=mock_code_workflow,
        full_pipeline_workflow=mock_full_workflow,
        cache=MagicMock(get=MagicMock(return_value=None), set=MagicMock()),
    ):
        from backend.main import app
        yield TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /preview-requirements ─────────────────────────────────────────────────────

def test_preview_requirements_success(client):
    r = client.post("/preview-requirements", json={"text": "POST /login accepts email and password. Returns 200 with token."})
    assert r.status_code == 200
    data = r.json()
    assert "overview" in data
    assert "quality_score" in data
    assert "_meta" in data


def test_preview_requirements_too_short(client):
    r = client.post("/preview-requirements", json={"text": "short"})
    assert r.status_code == 422


def test_preview_requirements_missing_text(client):
    r = client.post("/preview-requirements", json={})
    assert r.status_code == 422


# ── /analyze-requirements ─────────────────────────────────────────────────────

def test_analyze_requirements_success(client):
    r = client.post("/analyze-requirements", json={"text": "POST /login accepts email and password. Returns 200 with token."})
    assert r.status_code == 200
    data = r.json()
    assert "scenarios" in data
    assert isinstance(data["scenarios"], list)
    assert "_meta" in data


def test_analyze_requirements_returns_scenario_fields(client):
    r = client.post("/analyze-requirements", json={"text": "POST /login accepts email and password. Returns 200 with token."})
    sc = r.json()["scenarios"][0]
    assert "scenario_id" in sc
    assert "title" in sc


# ── /generate-test-cases ──────────────────────────────────────────────────────

SAMPLE_SCENARIO = {
    "scenario_id": "SCN-001",
    "title": "Valid login",
    "priority": "high",
    "type": "positive",
}


def test_generate_test_cases_success(client):
    r = client.post("/generate-test-cases", json={
        "scenarios": [SAMPLE_SCENARIO],
        "requirement_text": "POST /login accepts email and password.",
    })
    assert r.status_code == 200
    data = r.json()
    assert "test_cases" in data
    assert "applied_techniques" in data
    assert "_meta" in data


def test_generate_test_cases_empty_scenarios(client):
    r = client.post("/generate-test-cases", json={"scenarios": []})
    assert r.status_code == 200


def test_generate_test_cases_missing_body(client):
    r = client.post("/generate-test-cases", json={})
    assert r.status_code == 422


# ── /generate-automation-code ─────────────────────────────────────────────────

SAMPLE_TC = {
    "test_case_id": "TC-EP-001",
    "name": "Valid login",
    "scenario_id": "SCN-001",
    "technique": "EP",
    "steps": ["Setup", "POST /login", "Assert 200"],
    "expected_result": "200 OK",
    "priority": "high",
}


def test_generate_automation_code_success(client):
    r = client.post("/generate-automation-code", json={
        "test_cases": [SAMPLE_TC],
        "framework": "pytest",
    })
    assert r.status_code == 200
    data = r.json()
    assert "generated_files" in data
    assert data["framework"] == "pytest"
    assert "_meta" in data


def test_generate_automation_code_invalid_framework_still_works(client):
    r = client.post("/generate-automation-code", json={
        "test_cases": [SAMPLE_TC],
        "framework": "k6",
    })
    assert r.status_code == 200


def test_generate_automation_code_missing_test_cases(client):
    r = client.post("/generate-automation-code", json={"framework": "pytest"})
    assert r.status_code == 422


# ── /parse-file ───────────────────────────────────────────────────────────────

def test_parse_file_unsupported_type(client):
    r = client.post("/parse-file", files={"file": ("test.csv", b"a,b,c", "text/csv")})
    assert r.status_code == 400


def test_parse_file_txt(client):
    with patch("backend.main.extract_requirements", return_value="extracted text"):
        r = client.post("/parse-file", files={"file": ("req.txt", b"POST /login requires email and password.", "text/plain")})
        assert r.status_code == 200
        assert "content" in r.json()


def test_parse_file_empty_txt(client):
    with patch("backend.main.extract_requirements", return_value=""):
        r = client.post("/parse-file", files={"file": ("req.txt", b"   ", "text/plain")})
        assert r.status_code == 400


def test_parse_file_yaml_openapi(client):
    yaml_spec = b"""
openapi: 3.0.0
info:
  title: Payment API
  version: 1.0.0
paths:
  /payments:
    post:
      summary: Create payment
      responses:
        "201":
          description: Created
"""
    r = client.post("/parse-file", files={"file": ("spec.yaml", yaml_spec, "application/yaml")})
    assert r.status_code == 200
    data = r.json()
    assert "content" in data
    assert "Payment API" in data["content"]
    assert "POST /payments" in data["content"]


def test_parse_file_json_openapi(client):
    import json as _json
    spec = _json.dumps({
        "openapi": "3.0.0",
        "info": {"title": "Order API", "version": "1.0"},
        "paths": {
            "/orders": {
                "get": {
                    "summary": "List orders",
                    "responses": {"200": {"description": "OK"}}
                }
            }
        }
    }).encode()
    r = client.post("/parse-file", files={"file": ("spec.json", spec, "application/json")})
    assert r.status_code == 200
    data = r.json()
    assert "Order API" in data["content"]
    assert "GET /orders" in data["content"]


def test_parse_file_non_openapi_json_falls_back_to_text(client):
    plain_json = b'{"name": "foo", "description": "not a swagger spec"}'
    with patch("backend.main.extract_requirements", return_value="extracted fallback"):
        r = client.post("/parse-file", files={"file": ("data.json", plain_json, "application/json")})
    assert r.status_code == 200
    assert r.json()["content"] == "extracted fallback"


def test_parse_file_too_large(client):
    big = b"x" * (10 * 1024 * 1024 + 1)
    r = client.post("/parse-file", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413


def test_parse_file_yml_extension(client):
    yaml_spec = b"""
openapi: 3.0.0
info:
  title: Auth API
  version: 1.0.0
paths:
  /login:
    post:
      summary: Login
      responses:
        "200":
          description: OK
"""
    r = client.post("/parse-file", files={"file": ("spec.yml", yaml_spec, "application/yaml")})
    assert r.status_code == 200
    assert "Auth API" in r.json()["content"]
