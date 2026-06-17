import json
from pathlib import Path
from typing import Dict, Any
from backend.app.workflows.mvp_workflow import MVPWorkflow
from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow
from backend.app.workflows.automation_code_workflow import AutomationCodeWorkflow
from backend.app.schemas.agent_schemas import RequirementInput, RequirementAnalysisResult


class FullPipelineWorkflow:
    def __init__(self):
        self.requirement_workflow = MVPWorkflow()
        self.test_case_workflow = TestCasePayloadWorkflow()
        self.code_workflow = AutomationCodeWorkflow()
        self.output_dir = Path(__file__).resolve().parent.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_overview(self, analysis: RequirementAnalysisResult, preview: Dict[str, Any] = None) -> Dict[str, Any]:
        endpoints = list({s.related_endpoint for s in analysis.scenarios if s.related_endpoint})
        req_texts = [r.text for r in analysis.requirements_summary if r.text]
        ov = (preview or {}).get("overview", {})
        return {
            "summary": ov.get("summary") or (" ".join(req_texts[:2]) if req_texts else ""),
            "features": ov.get("features") or [],
            "endpoints": list({*endpoints, *ov.get("endpoints", [])}),
            "business_rules": ov.get("business_rules") or req_texts,
        }

    def run(self, payload: RequirementInput) -> Dict[str, Any]:
        framework = payload.framework or "pytest"

        # Get rich overview (summary, features, business_rules) from preview before analyze
        try:
            preview_data = self.requirement_workflow.analyzer.preview(payload.text)
        except Exception:
            preview_data = {}

        analysis = self.requirement_workflow.run(payload.text)
        overview = self._build_overview(analysis, preview=preview_data)

        test_cases_result = self.test_case_workflow.run(
            analysis.scenarios,
            requirement_text=payload.text,
            overview=overview,
        )
        automation_result = self.code_workflow.run(
            test_cases_result.test_cases,
            framework=framework,
            requirement_text=payload.text,
            payload_templates=test_cases_result.payload_templates,
            test_data_matrix=test_cases_result.test_data_matrix,
        )

        result = {
            "analysis": analysis.model_dump(),
            "test_cases_result": test_cases_result.model_dump(),
            "automation_result": automation_result,
        }

        self._save_artifacts(result)
        return result

    def _save_artifacts(self, result: Dict[str, Any]) -> None:
        json_path = self.output_dir / "full_pipeline_output.json"
        md_path = self.output_dir / "full_pipeline_output.md"

        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        md_lines = [
            "# Full Pipeline Output",
            "",
            "## 1. Requirement Analysis",
            f"- Scenarios: {len(result['analysis']['scenarios'])}",
            "",
            "## 2. Test Cases",
            f"- Test Cases: {len(result['test_cases_result']['test_cases'])}",
            "",
            "## 3. Automation Code",
            f"- Generated Files: {len(result['automation_result']['generated_files'])}",
            "",
            "## Generated Files",
        ]
        for file_info in result['automation_result']['generated_files']:
            md_lines.append(f"- {file_info['file_name']}")

        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
