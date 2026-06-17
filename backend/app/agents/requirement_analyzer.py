import json
import os
import re
from typing import Dict, Any

from openai import OpenAI, BadRequestError
from backend.app.services.groq_retry import call_with_backoff


def _sanitize_text(text: str) -> str:
    """Replace chars that cause LLM to produce broken JSON escape sequences."""
    return text.replace('\\', '/').replace('"', "'")

_PREVIEW_SYSTEM_PROMPT = """You are a senior QA analyst. Analyze the requirement and answer a checklist — do NOT compute scores yourself (the system calculates scores from your answers).

## Completeness checklist — answer true/false for each
- inputs_defined: Are input fields defined with types or formats?
- validation_rules: Are validation constraints specified (length, range, format, required/optional)?
- success_response: Is the success response or output explicitly defined?
- error_cases: Are error cases and HTTP status codes specified?
- business_rules: Are business rules and edge cases covered?
- auth_stated: Are authentication or permission requirements stated?

## Testability checklist — answer true/false for each
- acceptance_criteria: Are acceptance criteria measurable (pass/fail deterministic)?
- expected_outputs: Are expected outputs or responses explicitly stated?
- test_data_derivable: Can test data boundaries be derived (min/max, formats)?
- rules_verifiable: Does each rule map to at least one verifiable scenario?
- no_vague_language: Is the requirement free of "should/might/could" language?

## Clarity positive checklist — answer true/false for each
- has_concrete_examples: Does the requirement include concrete examples (sample values, sample requests/responses)?
- terms_defined: Are all domain-specific terms, acronyms, and technical concepts defined or explained?
- logical_flow: Is the requirement structured in a clear logical order (preconditions → action → result)?
- specific_acceptance_criteria: Do acceptance criteria use specific, measurable language (exact numbers, formats, status codes)?

## Clarity issues — list ONLY items that are objectively and clearly present
- vague_words: ONLY non-measurable technical qualifiers that prevent deterministic testing — e.g. "fast", "user-friendly", "appropriate", "good", "reasonable", "simple". Do NOT list modal verbs like "should/must/shall" (those are requirement conventions), common REST/HTTP terms, or plain English words. Be conservative: if unclear whether it is vague, do NOT include it.
- undefined_terms: ONLY domain-specific business acronyms or proprietary terms that a new engineer would not know without a definition. Do NOT list standard HTTP/REST/JSON/SQL terms, framework names, or common programming terms.
- conflicting_statements: ONLY pairs where two statements directly contradict each other (e.g. "field is required" vs "field is optional"). Do NOT list statements that are merely different or additive.
- implicit_assumptions: ONLY assumptions that materially affect test design and are NOT implied by the req type (e.g. "user must exist in DB" when no creation endpoint is given). Do NOT list obvious infrastructure assumptions ("server is running", "network is available").

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "overview": {
    "summary": "1-3 sentence plain-language summary",
    "features": ["feature or capability described"],
    "endpoints": ["HTTP method + path if present"],
    "business_rules": ["each distinct rule or constraint"]
  },
  "suggestions": [
    {
      "type": "missing",
      "title": "Short title",
      "description": "What is missing and why it matters for testing"
    }
  ],
  "quality_checks": {
    "completeness": {
      "inputs_defined": true,
      "validation_rules": false,
      "success_response": true,
      "error_cases": false,
      "business_rules": true,
      "auth_stated": false
    },
    "testability": {
      "acceptance_criteria": true,
      "expected_outputs": true,
      "test_data_derivable": false,
      "rules_verifiable": true,
      "no_vague_language": false
    },
    "clarity_positive": {
      "has_concrete_examples": true,
      "terms_defined": false,
      "logical_flow": true,
      "specific_acceptance_criteria": false
    },
    "clarity_issues": {
      "vague_words": ["appropriate"],
      "undefined_terms": [],
      "conflicting_statements": [],
      "implicit_assumptions": ["user is pre-registered"]
    }
  }
}
Only include suggestion types that actually apply. Keep descriptions concise (1-2 sentences).
For clarity_issues: list ONLY items you can quote or directly reference from the requirement — do not invent."""

_SCENARIOS_SYSTEM_PROMPT = """You are Requirement Analyzer, an expert QA and business analysis agent.

## Objective
Extract a comprehensive set of test scenarios from the provided requirement.

## Scenario types to cover — be thorough
- **positive / happy path**: normal successful flows
- **negative**: invalid input, missing fields, wrong types, unauthorized
- **boundary**: min/max values, empty strings, maximum lengths, zero
- **security**: SQL injection, XSS, auth bypass, privilege escalation, brute force
- **edge case**: race conditions, duplicate submissions, concurrent requests, special characters

## Instructions
1. Read ALL input carefully — do not skip any rule or constraint.
2. For EACH business rule and validation constraint explicitly stated in the requirement, create at least one scenario.
3. STRICT GROUNDING: Every scenario must trace directly to text present in the requirement. Do NOT invent endpoints, fields, rules, or behaviours not stated. If a common behaviour (e.g. 401) is not mentioned, put it in missing_information instead.
4. related_endpoint must be taken verbatim from the requirement. If no endpoint is stated, set related_endpoint to null.
5. List genuinely missing or ambiguous information in missing_information.

## ID Format Rules (strictly enforced)
- Requirement IDs: REQ-NNN — zero-padded 3 digits (REQ-001, REQ-002 … NOT REQ-1 or REQ001)
- Scenario IDs: SCN-NNN — zero-padded 3 digits (SCN-001, SCN-002 … NOT SCN-1)

## Field Constraints
- priority: MUST be exactly one of: "high", "medium", "low" (lowercase only)
- type: MUST be exactly one of: "positive", "negative", "boundary", "security", "edge case" (lowercase only)

## Output Format
Return ONLY a valid JSON object — no markdown, no explanation:
{
  "requirements_summary": [{"id": "REQ-001", "text": "requirement description"}],
  "scenarios": [
    {
      "scenario_id": "SCN-001",
      "title": "Register with valid payload",
      "description": "Verify successful user creation with all required fields",
      "given": "The /users endpoint is available and the email does not exist",
      "when": "POST /users is called with valid name, email, and password",
      "then": "Response is 201 with a user object containing id, name, email",
      "priority": "high",
      "type": "positive",
      "related_requirement": "REQ-001",
      "related_endpoint": "POST /users"
    }
  ],
  "missing_information": ["No rate-limiting spec provided", "Auth mechanism not specified"]
}"""


def _extract_json(text: str) -> str:
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")
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
    return text[start:]


def _repair_unescaped_quotes(text: str) -> str:
    """Repair JSON strings that contain literal quotes without escaping.

    This is a best-effort heuristic for malformed outputs where a model writes:
      {"summary": "Use "quotes" inside text"}
    and we need to recover to:
      {"summary": "Use \"quotes\" inside text"}
    """
    out = []
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            out.append(ch)
            escape_next = False
            continue

        if ch == '\\' and in_string:
            out.append(ch)
            escape_next = True
            continue

        if ch == '"':
            if in_string:
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j >= len(text) or text[j] in ',:}]':
                    out.append(ch)
                    in_string = False
                else:
                    out.append('\\"')
            else:
                k = len(out) - 1
                while k >= 0 and out[k] in " \t\r\n":
                    k -= 1
                prev_sig = out[k] if k >= 0 else ''
                if prev_sig in '{[,:':
                    out.append(ch)
                    in_string = True
                else:
                    # Unrecognized position — treat as opening quote anyway to avoid
                    # leaving in_string=False for the content that follows
                    out.append(ch)
                    in_string = True
            continue

        if in_string and ch in '\n\r':
            out.append('\\n' if ch == '\n' else '\\r')
            continue

        out.append(ch)

    return ''.join(out)



def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = _extract_json(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_unescaped_quotes(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Repair pass 2: add missing commas between values on separate lines.
            repaired2 = re.sub(
                r'([}]"0-9]|true|false|null)([ \t]*\n[ \t]*)("|\{|\[)',
                r'\1,\2\3',
                repaired,
            )
            return json.loads(repaired2)



_COMPLETENESS_ITEMS = {
    "inputs_defined":    ("Input fields defined with types/formats", 20),
    "validation_rules":  ("Validation rules & constraints specified", 20),
    "success_response":  ("Success response / output defined", 15),
    "error_cases":       ("Error cases & HTTP status codes specified", 20),
    "business_rules":    ("Business rules & edge cases covered", 15),
    "auth_stated":       ("Auth/permission requirements stated", 10),
}

_TESTABILITY_ITEMS = {
    "acceptance_criteria":  ("Acceptance criteria are measurable", 30),
    "expected_outputs":     ("Expected outputs explicitly stated", 25),
    "test_data_derivable":  ("Test data boundaries can be derived", 20),
    "rules_verifiable":     ("Each rule maps to a verifiable scenario", 15),
    "no_vague_language":    ("No vague language (should/might/could)", 10),
}

# Clarity: 8 binary items × 12.5 pts each = 100 max
# 4 positive (from LLM clarity_positive booleans)
# 4 negative (pass when the corresponding clarity_issues list is empty)
_CLARITY_ITEMS = {
    "has_concrete_examples":        ("Concrete examples provided",                   12.5),
    "terms_defined":                ("Domain terms & acronyms defined",              12.5),
    "logical_flow":                 ("Logical flow: precondition → action → result", 12.5),
    "specific_acceptance_criteria": ("Acceptance criteria use exact values/formats", 12.5),
    "no_vague_words":               ("No vague qualifiers",                          12.5),
    "no_undefined_terms":           ("No undefined terms/acronyms",                  12.5),
    "no_conflicting_statements":    ("No conflicting statements",                    12.5),
    "no_implicit_assumptions":      ("No implicit assumptions",                      12.5),
}

# Maps each negative item key → the issues list key it checks
_CLARITY_NEGATIVE_MAP = {
    "no_vague_words":            "vague_words",
    "no_undefined_terms":        "undefined_terms",
    "no_conflicting_statements": "conflicting_statements",
    "no_implicit_assumptions":   "implicit_assumptions",
}


def _compute_quality_score(data: Dict[str, Any]) -> Dict[str, Any]:
    checks = data.get("quality_checks", {})
    comp_checks  = checks.get("completeness", {})
    test_checks  = checks.get("testability", {})
    pos_checks   = checks.get("clarity_positive", {})
    clarity_issues = checks.get("clarity_issues", {})

    # Completeness: sum awarded points
    completeness = 0
    comp_found, comp_missing = [], []
    for key, (label, pts) in _COMPLETENESS_ITEMS.items():
        if comp_checks.get(key, False):
            completeness += pts
            comp_found.append(f"{label} (+{pts})")
        else:
            comp_missing.append(f"{label} (0/{pts})")

    # Testability: sum awarded points
    testability = 0
    test_found, test_missing = [], []
    for key, (label, pts) in _TESTABILITY_ITEMS.items():
        if test_checks.get(key, False):
            testability += pts
            test_found.append(f"{label} (+{pts})")
        else:
            test_missing.append(f"{label} (0/{pts})")

    # Clarity: 8 binary items × 12.5 pts each = 100 max
    ambiguity_raw = 0.0
    clarity_found, clarity_missing = [], []
    for key, (label, pts) in _CLARITY_ITEMS.items():
        if key in _CLARITY_NEGATIVE_MAP:
            items = clarity_issues.get(_CLARITY_NEGATIVE_MAP[key], [])
            if not isinstance(items, list):
                items = []
            passed = len(items) == 0
        else:
            passed = pos_checks.get(key, False)
        if passed:
            ambiguity_raw += pts
            clarity_found.append(f"{label} (+{int(pts)})")
        else:
            clarity_missing.append(f"{label} (0/{int(pts)})")
    ambiguity = round(ambiguity_raw)

    overall = round(completeness * 0.40 + testability * 0.35 + ambiguity * 0.25)
    risk = "High" if overall < 60 else ("Medium" if overall < 80 else "Low")

    return {
        "overall": overall,
        "completeness": completeness,
        "testability": testability,
        "ambiguity": ambiguity,
        "risk": risk,
        "score_breakdown": {
            "completeness_found":   comp_found,
            "completeness_missing": comp_missing,
            "testability_found":    test_found,
            "testability_missing":  test_missing,
            "clarity_found":        clarity_found,
            "clarity_missing":      clarity_missing,
        },
    }


_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_AIP_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"


class RequirementAnalyzer:
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

    def _chat(self, model: str, max_tokens: int, system: str, user: str) -> str:
        def _once(u=user):
            completion = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                seed=42,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
            )
            return completion.choices[0].message.content or ""

        try:
            return call_with_backoff(_once, label="Agent1")
        except BadRequestError as e:
            if "json_validate_failed" in str(e):
                # Retry without server-side JSON validation — rely on client-side _parse_json
                retry_user = "Output ONLY a valid JSON object, no markdown, no extra text.\n\n" + user
                def _retry_once(u=retry_user):
                    completion = self.client.chat.completions.create(
                        model=model, max_tokens=max_tokens, temperature=0, seed=42,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
                    )
                    return completion.choices[0].message.content or ""
                return call_with_backoff(_retry_once, label="Agent1-retry")
            raise RuntimeError(f"API bad request: {e}") from e

    def preview(self, text: str) -> Dict[str, Any]:
        safe = _sanitize_text(text)
        prompt = f"Analyze this requirement:\n\n{safe}"
        raw = self._chat(self.light_model, 3000, _PREVIEW_SYSTEM_PROMPT, prompt)
        try:
            data = _parse_json(raw)
        except json.JSONDecodeError:
            raw = self._chat(
                self.light_model, 3000, _PREVIEW_SYSTEM_PROMPT,
                "Output ONLY a valid JSON object, no markdown, no extra text.\n\n" + prompt,
            )
            data = _parse_json(raw)
        data["_input_length"] = len(safe)
        data["quality_score"] = _compute_quality_score(data)
        return data

    def analyze(self, text: str) -> Dict[str, Any]:
        safe = _sanitize_text(text)
        prompt = f"Generate comprehensive test scenarios for this requirement:\n\n{safe}"
        raw = self._chat(self.model, 6000, _SCENARIOS_SYSTEM_PROMPT, prompt)
        try:
            return _parse_json(raw)
        except json.JSONDecodeError:
            raw = self._chat(
                self.model, 6000, _SCENARIOS_SYSTEM_PROMPT,
                "Output ONLY a valid JSON object, no markdown, no extra text.\n\n" + prompt,
            )
            return _parse_json(raw)
