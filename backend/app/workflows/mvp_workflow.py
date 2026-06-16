from backend.app.agents.requirement_analyzer import RequirementAnalyzer
from backend.app.schemas.agent_schemas import RequirementAnalysisResult


class MVPWorkflow:
    def __init__(self):
        self.analyzer = RequirementAnalyzer()

    def run(self, text: str) -> RequirementAnalysisResult:
        raw_result = self.analyzer.analyze(text)
        raw_result.setdefault("scenarios", [])
        raw_result.setdefault("requirements_summary", [])
        raw_result.setdefault("missing_information", [])
        try:
            return RequirementAnalysisResult(**raw_result)
        except Exception:
            return RequirementAnalysisResult(
                requirements_summary=[],
                scenarios=[],
                missing_information=["Agent 1 returned an unexpected response format."],
            )
