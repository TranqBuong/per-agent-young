import json
import os
import re
from typing import List, Dict, Any, Optional

from groq import Groq, BadRequestError
from backend.app.services.groq_retry import call_with_backoff


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
    return {
        "test_case_id": f"TC-{sid.split('-')[-1].zfill(3)}",
        "name": (s.get("title") or sid)[:50],
        "scenario_id": sid,
        "technique": "UC",
        "preconditions": given,
        "steps": [given, when, f"Verify: {then}"],
        "test_data": {},
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
- Be selective: 3-6 techniques is typical; do not list a technique you cannot justify

Return ONLY valid JSON:
{
  "system_type": "traditional",
  "selected_techniques": [
    {
      "technique": "EP",
      "rationale": "Email and password fields have valid/invalid input partitions",
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
      "name": "Valid email login",
      "scenario_id": "SCN-001",
      "technique": "EP",
      "preconditions": "User exists, system available",
      "steps": ["POST /login with valid credentials", "Verify 200 response", "Check token in response"],
      "test_data": {"email": "user@test.com", "password": "Pass123!"},
      "expected_result": "200 OK with auth token",
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
            else:
                out.append(ch)
        return json.loads(''.join(out))


_DEFAULT_MODEL = "llama-3.1-8b-instant"


class TestCasePayloadGenerator:
    def __init__(self, model: str = None):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Export it before starting the server: export GROQ_API_KEY=<your-key>"
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("GROQ_MODEL", _DEFAULT_MODEL)
        self.light_model = os.environ.get("GROQ_MODEL_LIGHT", _DEFAULT_MODEL)

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
                raise ValueError(f"Empty response (finish_reason={completion.choices[0].finish_reason})")
            return _parse_json(raw)

        try:
            return call_with_backoff(_once, label="Agent2")
        except BadRequestError as e:
            if "json_validate_failed" in str(e):
                retry_user = "IMPORTANT: Output ONLY valid JSON.\n\n" + user
                return call_with_backoff(lambda: _once(retry_user), label="Agent2-retry")
            raise RuntimeError(f"Groq bad request: {e}") from e

    def generate(
        self,
        scenarios: List[Dict[str, Any]],
        requirement_text: Optional[str] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        # ── Call 1: select techniques ──────────────────────────────
        slim_scenarios = [
            {
                'scenario_id':       s.get('scenario_id', ''),
                'title':             (s.get('title') or '')[:80],
                'description':       (s.get('description') or '')[:100],
                'type':              s.get('type', ''),
                'priority':          s.get('priority', ''),
                'related_endpoint':  s.get('related_endpoint') or '',
                'given':             _trunc(s.get('given') or '', 200),
                'when':              _trunc(s.get('when') or '', 200),
                'then':              _trunc(s.get('then') or '', 200),
            }
            for s in scenarios
        ]

        context_parts = []
        if requirement_text:
            context_parts.append(f"Requirement:\n{_trunc(requirement_text, 1500)}")
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
        selected_techniques = selected_techniques[:5]
        technique_ids = [t["technique"] for t in selected_techniques]
        techniques_detail = "\n".join(
            f"- {t['technique']}: {(t.get('rationale') or '')[:80]}"
            for t in selected_techniques
        )

        scenario_ids = [s['scenario_id'] for s in slim_scenarios]
        req_context = f"Requirement context:\n{_trunc(requirement_text, 1500)}\n\n" if requirement_text else ""
        tc_user = (
            f"{req_context}"
            f"Techniques: {', '.join(technique_ids)}\n"
            f"Scenarios to cover: {', '.join(scenario_ids)}\n\n"
            f"Technique details:\n{techniques_detail}\n\n"
            f"Scenarios:\n{json.dumps(slim_scenarios, ensure_ascii=False)}"
        )

        tc_result = self._call(
            system=_TEST_CASE_GENERATION_PROMPT,
            user=tc_user,
            max_tokens=3500,
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
            tc["technique"] = tech  # normalize so technique field matches the rebuilt ID

        # Keep only techniques actually used
        used = {tc["technique"] for tc in test_cases}
        verified_techniques = [t for t in selected_techniques if t["technique"] in used]

        return {
            "system_type": system_type,
            "applied_techniques": verified_techniques,
            "test_cases": test_cases,
            "test_data_matrix": [],
            "payload_templates": [],
        }
