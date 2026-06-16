from typing import List, Dict, Any, Optional
from backend.app.agents.automation_code_writer import AutomationCodeWriter
from backend.app.schemas.agent_schemas import TestCaseItem


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
        return self.writer.generate(
            raw_test_cases,
            framework=framework,
            requirement_text=requirement_text,
            payload_templates=payload_templates,
            test_data_matrix=test_data_matrix,
        )
