from typing import List, Optional, Dict, Any
from backend.app.agents.test_case_payload_generator import TestCasePayloadGenerator, make_fallback_tc
from backend.app.schemas.agent_schemas import ScenarioItem, TestCaseItem, TestCasePayloadResult


class TestCasePayloadWorkflow:
    def __init__(self):
        self.generator = TestCasePayloadGenerator()

    def run(
        self,
        scenarios: List[ScenarioItem],
        requirement_text: Optional[str] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> TestCasePayloadResult:
        raw_result = self.generator.generate(
            [s.model_dump() for s in scenarios],
            requirement_text=requirement_text,
            overview=overview,
        )
        result = TestCasePayloadResult(**raw_result)
        self._validate_grounding(scenarios, result)
        self._build_payload_templates(result)
        return result

    def _validate_grounding(
        self, scenarios: List[ScenarioItem], result: TestCasePayloadResult
    ) -> None:
        valid_ids = {s.scenario_id for s in scenarios}

        # Remove test cases that reference a scenario_id not in input (hallucinated)
        before = len(result.test_cases)
        result.test_cases = [
            tc for tc in result.test_cases
            if tc.scenario_id is None or tc.scenario_id in valid_ids
        ]
        dropped = before - len(result.test_cases)
        if dropped:
            import warnings
            warnings.warn(f"Grounding check: dropped {dropped} test case(s) with invalid scenario_id")

        # Ensure every input scenario has at least one test case
        covered = {tc.scenario_id for tc in result.test_cases}
        for s in scenarios:
            if s.scenario_id not in covered:
                fallback = make_fallback_tc(s.model_dump(), len(result.test_cases))
                result.test_cases.append(TestCaseItem(**fallback))

    def _build_payload_templates(self, result: "TestCasePayloadResult") -> None:
        """Build payload_templates deterministically from test_data if LLM left it empty."""
        if result.payload_templates:
            return
        templates = []
        for tc in result.test_cases:
            if tc.test_data:
                templates.append({
                    "test_case_id": tc.test_case_id,
                    "scenario_id": tc.scenario_id,
                    "name": tc.name,
                    "payload": tc.test_data,
                    "technique": tc.technique,
                    "priority": tc.priority,
                })
        result.payload_templates = templates
