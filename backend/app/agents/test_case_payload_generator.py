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


_TEST_CASE_GENERATION_PROMPT = """You are a senior QA engineer. Your goal: MAXIMIZE test coverage for each scenario by generating MULTIPLE test cases per scenario.

## How to analyze each scenario
For each scenario, examine its "given/when/then" and think through ALL relevant test angles:
1. **Happy path** (EP) — valid input, expected success
2. **Boundary values** (BVA) — if the scenario involves numeric ranges, string lengths, dates, or limits: test min, max, min-1, max+1
3. **Invalid / negative input** (EG) — missing fields, wrong types, out-of-range, unauthorized access, empty values
4. **Security** (EG) — if input fields accept user text: XSS, SQL injection, special characters
5. **Edge cases** (EP/EG) — empty list, zero, null, duplicate, very long string

Generate a SEPARATE test case for EACH test angle that is relevant to the scenario. A simple scenario may produce 2-3 TCs; a complex input scenario may produce 4-6 TCs.

## Grounding rules (strictly enforced)
- expected_result MUST be derived from the scenario's "then" field — paraphrase it, do NOT invent a different outcome
- steps[1] (the action step) MUST reflect the scenario's "when" field
- steps[0] (setup) MUST reflect the scenario's "given" field
- test_data values MUST use field names stated in the requirement — do NOT invent fields
- Do NOT add test cases for scenarios not in the input list

## Coverage rules
- EVERY scenario_id MUST have MULTIPLE test cases (at minimum 2: one positive + one negative/edge)
- Every listed technique MUST appear in at least one test_case across all scenarios
- scenario_id in each test case MUST exactly match one of the input scenario_ids

## Format rules
- name ≤10 words, preconditions ≤15 words, steps exactly 3 short strings, expected_result ≤15 words
- test_data values MUST be literal strings or numbers — no code, no expressions
- For boundary values, write the actual literal value (no .repeat())
- For XSS/SQL, write plain JSON strings: "<script>alert(1)</script>", "' OR 1=1 --"
- test_case_id format: TC-TECHNIQUE-NNN (e.g. TC-EP-001, TC-BVA-002)
- priority: exactly "high", "medium", or "low"

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
    # JSON was truncated — close all open structures
    partial = text[start:]
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

    def _call(
        self, system: str, user: str, max_tokens: int,
        light: bool = False, use_json_format: bool = True,
    ) -> Dict[str, Any]:
        model = self.light_model if light else self.model

        def _once(u=user):
            kwargs = dict(model=model, max_tokens=max_tokens, temperature=0, seed=42,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": u}])
            if use_json_format:
                # GreenNode bug: response_format + max_tokens >= ~2000 → empty content.
                # Only use json_object mode when max_tokens is safely below the threshold.
                kwargs["response_format"] = {"type": "json_object"}
            completion = self.client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content or ""
            if not raw.strip():
                finish = completion.choices[0].finish_reason
                if finish == "length":
                    raise json.JSONDecodeError(f"Empty response (finish_reason=length)", "", 0)
                raise ValueError(f"Empty response (finish_reason={finish})")
            return _parse_json(raw)

        try:
            return call_with_backoff(_once, label="Agent2")
        except (BadRequestError, json.JSONDecodeError) as e:
            if isinstance(e, BadRequestError) and "json_validate_failed" not in str(e):
                raise RuntimeError(f"API bad request: {e}") from e
            # Retry without response_format regardless of the cause
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

        # ── Technique selection: rule-based heuristic (no LLM call) ──────────
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

        scenario_ids_all = [s['scenario_id'] for s in slim_scenarios]
        types_present = {s.get('type', '').lower() for s in slim_scenarios}

        _TYPE_TECHNIQUE_MAP = [
            ({'positive', 'edge case'},          'EP',  'equivalence partitioning'),
            ({'boundary'},                        'BVA', 'boundary value analysis'),
            ({'negative', 'security'},            'EG',  'error guessing'),
        ]
        selected_techniques = []
        for trigger_types, technique, rationale in _TYPE_TECHNIQUE_MAP:
            if types_present & trigger_types and len(selected_techniques) < 4:
                selected_techniques.append({
                    "technique": technique,
                    "rationale": rationale,
                    "applicable_scenarios": scenario_ids_all,
                })
        # UC: add only when multiple distinct scenario types are present (multi-step flows)
        if len(types_present) >= 2 and len(selected_techniques) < 4:
            selected_techniques.append({
                "technique": "UC",
                "rationale": "use case testing — multi-type scenarios detected",
                "applicable_scenarios": scenario_ids_all,
            })
        if not selected_techniques:
            selected_techniques = [
                {"technique": "EP", "rationale": "default", "applicable_scenarios": scenario_ids_all},
                {"technique": "EG", "rationale": "default", "applicable_scenarios": scenario_ids_all},
            ]

        system_type = "traditional"

        # ── Call 2: generate test cases (batched) ────────────────────
        technique_ids = [t["technique"] for t in selected_techniques]
        techniques_detail = "\n".join(
            f"- {t['technique']}: {(t.get('rationale') or '')[:60]}"
            for t in selected_techniques
        )

        # Sort by priority; batch 8 scenarios at a time so each batch produces
        # 8×M TCs (~32) — well within the 4000-token output budget.
        _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        sorted_scenarios = sorted(
            slim_scenarios, key=lambda s: _PRIORITY_ORDER.get(s.get("priority", "low"), 2)
        )
        _BATCH_SIZE_TC = 8
        batches = [sorted_scenarios[i:i + _BATCH_SIZE_TC] for i in range(0, len(sorted_scenarios), _BATCH_SIZE_TC)]

        ov_header = f"{overview_ctx}\n\n" if overview_ctx else ""
        req_context = f"Requirement:\n{_trunc(requirement_text, 2500)}\n\n" if requirement_text else ""

        test_cases: List[Dict[str, Any]] = []
        for batch in batches:
            scenario_ids = [s['scenario_id'] for s in batch]
            tc_user = (
                f"{ov_header}"
                f"{req_context}"
                f"Techniques: {', '.join(technique_ids)}\n"
                f"Scenarios to cover: {', '.join(scenario_ids)}\n\n"
                f"Technique details:\n{techniques_detail}\n\n"
                f"Scenarios:\n{json.dumps(batch, ensure_ascii=False)}"
            )
            # use_json_format=False: avoids GreenNode empty-content bug above ~1800 tokens.
            # max_tokens=4000: 8 scenarios × 4 techniques = ~32 TCs — need headroom.
            batch_result = self._call(
                system=_TEST_CASE_GENERATION_PROMPT,
                user=tc_user,
                max_tokens=4000,
                use_json_format=False,
            )
            test_cases.extend(batch_result.get("test_cases", []))

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
