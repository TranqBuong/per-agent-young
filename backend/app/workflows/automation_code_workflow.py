import warnings
from typing import List, Dict, Any, Optional
from backend.app.agents.automation_code_writer import AutomationCodeWriter
from backend.app.schemas.agent_schemas import TestCaseItem, AutomationCodeResult, GeneratedCodeFile


class AutomationCodeWorkflow:
    def __init__(self):
        self.writer = AutomationCodeWriter()

    def run(
        self,
        test_cases: List[TestCaseItem],
        framework: str = "pytest",
        requirement_text: Optional[str] = None,
        payload_templates: Optional[List[Dict[str, Any]]] = None,
        test_data_matrix: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        raw_test_cases = [tc.model_dump() for tc in test_cases]
        raw = self.writer.generate(
            raw_test_cases,
            framework=framework,
            requirement_text=requirement_text,
            payload_templates=payload_templates,
            test_data_matrix=test_data_matrix,
        )
        # Drop files missing required fields before schema validation
        valid_files = []
        for f in raw.get("generated_files", []):
            if f.get("file_name") and f.get("code"):
                valid_files.append(f)
            else:
                warnings.warn(f"Agent 3: dropping file missing file_name or code: {f.get('file_name', '<no name>')}")
        raw["generated_files"] = valid_files
        result = AutomationCodeResult(**raw)
        return result.model_dump()
