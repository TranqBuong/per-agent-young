import json
import re
import warnings
from typing import List, Optional, Dict, Any
from backend.app.agents.test_case_payload_generator import TestCasePayloadGenerator, make_fallback_tc
from backend.app.schemas.agent_schemas import ScenarioItem, TestCaseItem, TestCasePayloadResult


_DEDUP_SYSTEM_PROMPT = """You are a senior QA reviewer. Analyze the following test cases for semantic duplicates.

Two test cases are DUPLICATES if they test the same scenario with the same conditions and same expected outcome — regardless of which technique generated them or minor wording differences in the name.

For each group of duplicates, keep the RICHEST one (most steps, most test_data keys, highest priority). If equal, keep the first in the list.

Return ONLY valid JSON, no markdown:
{
  "keep": ["TC-EP-001", "TC-BVA-002"],
  "removed": [
    {"id": "TC-EP-002", "reason": "same scenario and expected result as TC-EP-001"}
  ]
}

Every input ID MUST appear in exactly one of "keep" or "removed"."""


def _tc_compact(tc: TestCaseItem) -> Dict[str, Any]:
    return {
        "id": tc.test_case_id,
        "name": tc.name,
        "scenario_id": tc.scenario_id,
        "technique": tc.technique,
        "expected_result": tc.expected_result or "",
        "steps": tc.steps or [],
        "test_data_keys": list((tc.test_data or {}).keys()),
        "priority": tc.priority or "medium",
    }


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _detail_score(tc: TestCaseItem) -> int:
    return (
        len(tc.steps or []) * 2
        + len(tc.test_data or {}) * 3
        + {"high": 2, "medium": 1, "low": 0}.get(tc.priority or "", 0)
        + len(tc.name or "") // 10
    )


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
        self._deduplicate_test_cases(result)
        self._ensure_minimum_test_case_count(scenarios, result)
        self._renumber_test_cases(result)
        self._build_payload_templates(result)
        return result

    def _validate_grounding(
        self, scenarios: List[ScenarioItem], result: TestCasePayloadResult
    ) -> None:
        valid_ids = {s.scenario_id for s in scenarios}

        # Remove test cases whose scenario_id is missing or not in the input set (hallucinated)
        before = len(result.test_cases)
        result.test_cases = [
            tc for tc in result.test_cases
            if tc.scenario_id and tc.scenario_id in valid_ids
        ]
        dropped = before - len(result.test_cases)
        if dropped:
            warnings.warn(f"Grounding check: dropped {dropped} test case(s) with invalid scenario_id")

        # Ensure every input scenario has at least one test case
        covered = {tc.scenario_id for tc in result.test_cases}
        for s in scenarios:
            if s.scenario_id not in covered:
                fallback = make_fallback_tc(s.model_dump(), len(result.test_cases))
                result.test_cases.append(TestCaseItem(**fallback))

    def _deduplicate_test_cases(self, result: TestCasePayloadResult) -> None:
        """String-fingerprint dedup — fast, no extra LLM call."""
        if len(result.test_cases) < 2:
            return
        self._deduplicate_test_cases_string(result)

    def _deduplicate_test_cases_ai(self, result: TestCasePayloadResult) -> None:
        """Ask the LLM to identify semantic duplicates across all test cases."""
        tc_list = [_tc_compact(tc) for tc in result.test_cases]
        user_msg = (
            "Review these test cases for semantic duplicates:\n\n"
            + json.dumps(tc_list, ensure_ascii=False, indent=2)
        )
        response = self.generator._call(_DEDUP_SYSTEM_PROMPT, user_msg, max_tokens=800, light=True)

        keep_ids: List[str] = response.get("keep", [])
        removed_list: List[Dict] = response.get("removed", [])

        all_ids = {tc.test_case_id for tc in result.test_cases}

        # Validate: every returned ID must actually exist
        unknown_keep = [i for i in keep_ids if i not in all_ids]
        unknown_removed = [r["id"] for r in removed_list if r.get("id") not in all_ids]
        if unknown_keep or unknown_removed:
            raise ValueError(
                f"LLM returned unknown IDs — keep:{unknown_keep} removed:{unknown_removed}"
            )

        # Validate: every input ID accounted for
        returned_ids = set(keep_ids) | {r["id"] for r in removed_list}
        missing = all_ids - returned_ids
        if missing:
            raise ValueError(f"LLM did not account for IDs: {missing}")

        if not removed_list:
            return  # no duplicates found

        remove_ids = {r["id"] for r in removed_list}
        result.test_cases = [tc for tc in result.test_cases if tc.test_case_id not in remove_ids]

        for r in removed_list:
            warnings.warn(
                f"Dedup (AI): removed {r['id']} — {r.get('reason', 'duplicate')}"
            )

    def _deduplicate_test_cases_string(self, result: TestCasePayloadResult) -> None:
        """String-fingerprint fallback dedup (scenario_id + normalized name + expected_result)."""
        seen: Dict[str, TestCaseItem] = {}
        kept: List[TestCaseItem] = []
        removed = 0

        for tc in result.test_cases:
            fp = (
                (tc.scenario_id or "")
                + "|" + (tc.technique or "")
                + "|" + _norm(tc.name)
                + "|" + _norm(tc.expected_result or "")
            )
            if fp not in seen:
                seen[fp] = tc
                kept.append(tc)
            else:
                existing = seen[fp]
                if _detail_score(tc) > _detail_score(existing):
                    idx = kept.index(existing)
                    kept[idx] = tc
                    seen[fp] = tc
                removed += 1

        if removed:
            warnings.warn(f"Dedup: removed {removed} duplicate test case(s) ({len(kept)} kept)")
        result.test_cases = kept

    def _ensure_minimum_test_case_count(self, scenarios: List[ScenarioItem], result: TestCasePayloadResult) -> None:
        """Ensure we still have at least one test case for every input scenario."""
        covered = {tc.scenario_id for tc in result.test_cases if tc.scenario_id}
        for s in scenarios:
            if s.scenario_id not in covered:
                result.test_cases.append(TestCaseItem(**make_fallback_tc(s.model_dump(), len(result.test_cases))))

    def _renumber_test_cases(self, result: TestCasePayloadResult) -> None:
        """Guarantee TC-TECHNIQUE-NNN format for ALL TCs including workflow-injected fallbacks."""
        tech_counters: Dict[str, int] = {}
        for tc in result.test_cases:
            tech = (tc.technique or "UC").upper()
            tc.technique = tech
            tech_counters[tech] = tech_counters.get(tech, 0) + 1
            tc.test_case_id = f"TC-{tech}-{tech_counters[tech]:03d}"

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
