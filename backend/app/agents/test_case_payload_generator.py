import json
import logging
import os
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI, BadRequestError
from backend.app.services.groq_retry import call_with_backoff

_logger = logging.getLogger(__name__)


def _trunc(text: str, limit: int) -> str:
    """Truncate at word boundary and append marker so LLM knows content was cut."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(' ', 1)[0]
    return cut + " [... truncated]"


def make_fallback_tc(s: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Single canonical fallback test case builder — used by both generator and workflow."""
    sid = s.get("scenario_id", f"SCN-{index+1:03d}")
    given = (s.get("given") or "System is available").strip() or "System is available"
    when  = (s.get("when")  or "Perform the action").strip()  or "Perform the action"
    then  = (s.get("then")  or "Outcome matches specification").strip() or "Outcome matches specification"
    # Build minimal test_data so Agent 3 has something to work with (E6 fix)
    test_data: Dict[str, Any] = {}
    endpoint = (s.get("related_endpoint") or "").strip()
    if endpoint:
        test_data["endpoint"] = endpoint
    generic_when = ("Perform the action", "")
    if when not in generic_when:
        test_data["action"] = when[:100]
    return {
        "test_case_id": f"TC-UC-{sid.split('-')[-1].zfill(3)}",
        "name": (s.get("title") or sid)[:50],
        "scenario_id": sid,
        "technique": "UC",
        "preconditions": given,
        "steps": [given, when, f"Verify: {then}"],
        "test_data": test_data,
        "expected_result": then,
        "priority": s.get("priority", "medium"),
        "tags": [s.get("type", "positive"), "uc"],
    }

_TECHNIQUE_SELECTION_PROMPT = """You are a QA expert. Given a requirement and its scenarios, select the test design techniques that MUST be applied for thorough coverage.

Available techniques:
Traditional: EP(equivalence partitioning), BVA(boundary value analysis), DT(decision table), ST(state transition), UC(use case testing), EG(error guessing)
AI/ML systems: FA(functional accuracy), RT(robustness testing), CT(consistency testing), HD(hallucination detection), BF(bias & fairness), OV(output validation), BP(boundary probing)

Rules:
- Classify system as "traditional", "ai", or "hybrid"
- Select ONLY techniques that genuinely apply to this requirement
- For each technique, specify which scenario IDs it applies to
- Use each scenario's related_requirement and related_endpoint when available to ground your selection
- Be selective: 3-6 techniques is typical; do not list a technique you cannot justify

Return ONLY valid JSON:
{
  "system_type": "traditional",
  "selected_techniques": [
    {
      "technique": "EP",
      "applicable_scenarios": ["SCN-001", "SCN-003"]
    }
  ]
}"""

_TEST_CASE_GENERATION_PROMPT = """You are a QA engineer. Apply EVERY listed technique to the scenarios. Be CONCISE.

## Grounding rules (strictly enforced)
- expected_result MUST be derived from the scenario's "then" field — paraphrase it, do NOT invent a different outcome
- steps[1] (the action step) MUST reflect the scenario's "when" field — do not change the action
- steps[0] (setup) MUST reflect the scenario's "given" field
- test_data values MUST use field names and constraints stated in the requirement — do not invent fields not mentioned
- Do NOT add test cases for scenarios not in the input list

## Coverage rules
- EVERY scenario_id MUST appear in at least one test case
- Total test_cases >= total scenarios
- Every technique MUST appear in at least one test_case
- scenario_id in each test case MUST exactly match one of the input scenario_ids
- If a scenario includes related_endpoint, reflect that endpoint in the action step or expected outcome when appropriate

## Format rules
- Keep fields SHORT: name ≤10 words, preconditions ≤15 words, steps exactly 3 short strings, expected_result ≤15 words
- test_data values MUST be literal strings or numbers — NEVER JavaScript expressions or code (no new Array(), no +, no functions)
- For long boundary values, write the actual literal value directly (no .repeat())
- For special chars like XSS/SQL, write them as plain JSON strings: "<script>alert(1)</script>", "' OR 1=1 --"
- test_case_id MUST follow format TC-TECHNIQUE-NNN (e.g. TC-EP-001, TC-BVA-002) — ALWAYS include the technique abbreviation, zero-padded 3-digit number
- priority MUST be exactly one of: "high", "medium", "low" (lowercase only)

Return ONLY valid JSON:
{
  "test_cases": [
    {
      "test_case_id": "TC-EP-001",
      "name": "Valid input returns success",
      "scenario_id": "SCN-001",
      "technique": "EP",
      "preconditions": "System available, required resources exist",
      "steps": ["Prepare valid input data per requirement", "Send request to the target endpoint", "Verify response matches expected outcome"],
      "test_data": {"field": "valid_value"},
      "expected_result": "Success response with expected payload",
      "priority": "high",
      "tags": ["positive", "ep"]
    }
  ]
}"""


def _extract_json(text: str) -> str:
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth, in_string, escape_next = 0, False, False
    last_valid_end = -1
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False; continue
        if ch == '\\' and in_string:
            escape_next = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if not in_string:
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
                if depth == 1:
                    last_valid_end = i  # last closed inner object
    # JSON was truncated — close all open structures
    partial = text[start:]
    # Count unclosed brackets
    opens = partial.count('{') - partial.count('}')
    arrays = partial.count('[') - partial.count(']')
    # Strip trailing incomplete item (find last complete '}')
    last_brace = partial.rfind('}')
    if last_brace != -1:
        partial = partial[:last_brace+1]
        # Close open arrays and objects
        partial += ']' * max(0, partial.count('[') - partial.count(']'))
        partial += '}' * max(0, partial.count('{') - partial.count('}'))
        return partial
    raise ValueError("Could not extract valid JSON")


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = _extract_json(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair pass 1: escape unescaped control characters inside strings
        out, in_string, escape_next = [], False, False
        for ch in text:
            if escape_next:
                out.append(ch); escape_next = False
            elif ch == '\\':
                out.append(ch); escape_next = True
            elif ch == '"':
                out.append(ch); in_string = not in_string
            elif in_string and ch == '\n':
                out.append('\\n')
            elif in_string and ch == '\r':
                out.append('\\r')
            elif in_string and ch == '\t':
                out.append('\\t')
            elif in_string and ord(ch) < 0x20:
                out.append(f'\\u{ord(ch):04x}')
            else:
                out.append(ch)
        repaired = ''.join(out)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Repair pass 2: add missing commas between values on separate lines.
            # After pass 1, all newlines inside strings are escaped, so remaining
            # newlines are structural — safe to insert commas before next tokens.
            repaired2 = re.sub(
                r'([\]}"0-9]|true|false|null)([ \t]*\n[ \t]*)("|\{|\[)',
                r'\1,\2\3',
                repaired,
            )
            return json.loads(repaired2)


_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_AIP_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"


class TestCasePayloadGenerator:
    def __init__(self, model: str = None):
        api_key = os.environ.get("GREENNODE_AIP_KEY")
        if not api_key:
            raise RuntimeError(
                "GREENNODE_AIP_KEY environment variable is not set. "
                "Export it before starting the server: export GREENNODE_AIP_KEY=<your-key>"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("GREENNODE_AIP_BASE_URL", _AIP_BASE_URL),
        )
        self.model = model or os.environ.get("GREENNODE_MODEL", _DEFAULT_MODEL)
        self.light_model = os.environ.get("GREENNODE_MODEL_LIGHT", _DEFAULT_MODEL)

    def _call(self, system: str, user: str, max_tokens: int, light: bool = False) -> Dict[str, Any]:
        model = self.light_model if light else self.model

        def _once(u=user):
            completion = self.client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0, seed=42,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
            )
            raw = completion.choices[0].message.content or ""
            if not raw.strip():
                finish = completion.choices[0].finish_reason
                if finish == "length":
                    # response_format + truncation → empty content from GreenNode; fall to retry without response_format
                    raise json.JSONDecodeError(f"Empty response (finish_reason=length)", "", 0)
                raise ValueError(f"Empty response (finish_reason={finish})")
            return _parse_json(raw)

        try:
            return call_with_backoff(_once, label="Agent2")
        except (BadRequestError, json.JSONDecodeError) as e:
            if isinstance(e, BadRequestError) and "json_validate_failed" not in str(e):
                raise RuntimeError(f"API bad request: {e}") from e
            # json_validate_failed OR finish_reason=length with empty content
            # → retry without response_format, same user message (don't grow input)
            def _retry_once(u=user):
                completion = self.client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=0, seed=42,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
                )
                raw = completion.choices[0].message.content or ""
                finish = completion.choices[0].finish_reason
                if not raw.strip():
                    _logger.warning("Agent2-retry empty content finish_reason=%s — returning empty TCs", finish)
                    return {"test_cases": []}
                return _parse_json(raw)
            return call_with_backoff(_retry_once, label="Agent2-retry")

    def generate(
        self,
        scenarios: List[Dict[str, Any]],
        requirement_text: Optional[str] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        requirement_text = requirement_text.replace('\\', '/').replace('"', "'") if requirement_text else requirement_text

        # Build compact overview context from Agent 1 output (E1 fix)
        overview_ctx = ""
        if overview:
            ov_parts = []
            endpoints = overview.get("endpoints") or []
            rules = overview.get("business_rules") or []
            if endpoints:
                ov_parts.append("Endpoints: " + ", ".join(str(ep)[:60] for ep in endpoints[:6]))
            if rules:
                ov_parts.append("Business rules: " + "; ".join(str(r)[:80] for r in rules[:5]))
            if ov_parts:
                overview_ctx = "API Overview:\n" + "\n".join(ov_parts)

        # ── Call 1: select techniques ──────────────────────────────
        slim_scenarios = [
            {
                'scenario_id':       s.get('scenario_id', ''),
                'title':             (s.get('title') or '')[:80],
                'description':       (s.get('description') or '')[:100],
                'type':              s.get('type', ''),
                'priority':          s.get('priority', ''),
                'related_requirement': s.get('related_requirement') or '',
                'related_endpoint':  s.get('related_endpoint') or '',
                'given':             _trunc(s.get('given') or '', 500),
                'when':              _trunc(s.get('when') or '', 500),
                'then':              _trunc(s.get('then') or '', 500),
            }
            for s in scenarios
        ]

        context_parts = []
        if overview_ctx:
            context_parts.append(overview_ctx)
        if requirement_text:
            context_parts.append(f"Requirement:\n{_trunc(requirement_text, 3200)}")
        context_parts.append(f"Scenarios:\n{json.dumps(slim_scenarios, ensure_ascii=False, indent=2)}")

        selection_result = self._call(
            system=_TECHNIQUE_SELECTION_PROMPT,
            user="Select applicable test design techniques for:\n\n" + "\n\n".join(context_parts),
            max_tokens=600,
            light=True,
        )

        selected_techniques = selection_result.get("selected_techniques", [])
        system_type = selection_result.get("system_type", "traditional")

        if not selected_techniques:
            scenario_ids_all = [s["scenario_id"] for s in slim_scenarios]
            selected_techniques = [
                {"technique": "EP", "rationale": "Default equivalence partitioning", "applicable_scenarios": scenario_ids_all},
                {"technique": "EG", "rationale": "Default error guessing", "applicable_scenarios": scenario_ids_all},
            ]

        # ── Call 2: generate test cases ──────────────────────────────
        selected_techniques = selected_techniques[:4]
        technique_ids = [t["technique"] for t in selected_techniques]
        techniques_detail = "\n".join(
            f"- {t['technique']}: {(t.get('rationale') or '')[:60]}"
            for t in selected_techniques
        )

        # Limit to 10 scenarios for TC generation — fallback covers the rest
        _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        slim_scenarios_tc = sorted(
            slim_scenarios, key=lambda s: _PRIORITY_ORDER.get(s.get("priority", "low"), 2)
        )[:10]

        scenario_ids = [s['scenario_id'] for s in slim_scenarios_tc]
        ov_header = f"{overview_ctx}\n\n" if overview_ctx else ""
        req_context = f"Requirement:\n{_trunc(requirement_text, 1500)}\n\n" if requirement_text else ""
        tc_user = (
            f"{ov_header}"
            f"{req_context}"
            f"Techniques: {', '.join(technique_ids)}\n"
            f"Scenarios to cover: {', '.join(scenario_ids)}\n\n"
            f"Technique details:\n{techniques_detail}\n\n"
            f"Scenarios:\n{json.dumps(slim_scenarios_tc, ensure_ascii=False)}"
        )

        tc_result = self._call(
            system=_TEST_CASE_GENERATION_PROMPT,
            user=tc_user,
            max_tokens=1800,
        )

        test_cases = tc_result.get("test_cases", [])

        # Ensure every scenario has at least one test case (fallback)
        covered_scenarios = {tc.get("scenario_id") for tc in test_cases if tc.get("scenario_id")}
        for s in slim_scenarios:
            if s["scenario_id"] not in covered_scenarios:
                test_cases.append(make_fallback_tc(s, len(test_cases)))

        # Normalize every test case — fill in missing required fields
        for i, tc in enumerate(test_cases):
            if not tc.get("name"):
                tc["name"] = tc.get("scenario_id", f"Test case {i+1}")
            if not isinstance(tc.get("steps"), list):
                tc["steps"] = [str(tc["steps"])] if tc.get("steps") else ["Execute the action", "Verify the result"]
            if not isinstance(tc.get("test_data"), dict):
                tc["test_data"] = {"value": str(tc["test_data"])} if tc.get("test_data") else {}
            if not isinstance(tc.get("tags"), list):
                tc["tags"] = [str(tc["tags"])] if tc.get("tags") else []
            tc.setdefault("preconditions", "System is available")
            tc.setdefault("expected_result", "Outcome matches specification")
            tc.setdefault("priority", "medium")

        # Rebuild ALL IDs to guarantee TC-TECHNIQUE-NNN format; normalize technique field too
        tech_counters: Dict[str, int] = {}
        for tc in test_cases:
            tech = (tc.get("technique") or "UC").upper()
            tech_counters[tech] = tech_counters.get(tech, 0) + 1
            tc["test_case_id"] = f"TC-{tech}-{tech_counters[tech]:03d}"
            tc["technique"] = tech

        # Build actual scenario coverage per technique from generated test_cases
        tech_scenarios: Dict[str, List[str]] = {}
        for tc in test_cases:
            tech = tc.get("technique", "")
            sid = tc.get("scenario_id", "")
            if tech and sid and sid not in tech_scenarios.get(tech, []):
                tech_scenarios.setdefault(tech, []).append(sid)

        # Keep all actually-used techniques; include any the LLM introduced beyond the selection
        used = {tc["technique"] for tc in test_cases}
        selected_tech_ids = {t["technique"] for t in selected_techniques}
        verified_techniques = []
        for t in selected_techniques:
            if t["technique"] in used:
                entry = dict(t)
                entry["applicable_scenarios"] = tech_scenarios.get(t["technique"], [])
                verified_techniques.append(entry)
        for tech in used - selected_tech_ids:
            verified_techniques.append({
                "technique": tech,
                "rationale": "applied by LLM",
                "applicable_scenarios": tech_scenarios.get(tech, []),
            })

        return {
            "system_type": system_type,
            "applied_techniques": verified_techniques,
            "test_cases": test_cases,
            "test_data_matrix": [],
            "payload_templates": [],
        }
