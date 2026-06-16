import json
import os
import re
from typing import List, Dict, Any, Optional

from groq import Groq, BadRequestError
from backend.app.services.groq_retry import call_with_backoff

_SYSTEM_PROMPT = """You are Automation Code Writer, an expert test automation engineer.

## Objective
Convert test cases into complete, executable automation code files.

## Framework guidance
- **pytest**: use `requests` library, `conftest.py` with fixtures (base_url, auth headers), clear assertions.
- **playwright**: use `playwright-python` sync API, `pytest-playwright` fixtures.
- **k6**: use k6 JavaScript API (`import http from 'k6/http'`), `check()` assertions, export `default function`.
- **selenium**: use `selenium` WebDriver with `pytest`, `webdriver.Chrome()` with `Options`, explicit `WebDriverWait`, `By` selectors.
- **postman**: generate a valid Postman Collection v2.1 JSON (`info`, `item[]` array). Each test case becomes one `item` with `request` (method, url, headers, body) and `event` (test scripts using `pm.test` / `pm.response.to.have.status`). Output as a single `.json` collection file.

## Grounding rules (strictly enforced)
1. Assertions MUST verify exactly what `expected_result` states — status code, response fields, or error messages stated there.
2. Use the exact values from `test_data` field — do not substitute or invent different values.
3. steps define the action sequence — follow them exactly.

## Implementation rules
4. Generate one test file per test case (or group related ones by scenario_id).
5. Each file must be complete and runnable — no `...` or TODO stubs.
6. Use environment variable references (`os.environ.get("BASE_URL", "http://localhost:8000")`).
7. Name each test function after test_case_id: e.g. `def test_TC001_valid_login():`.

## Output Format — REQUIRED
Use ONLY this separator format (no JSON, no markdown):

===FILE:tests/test_example.py===
import os
import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

def test_TC001_example():
    r = requests.post(f"{BASE_URL}/login", json={"email": "user@test.com"})
    assert r.status_code == 200
===END===

Repeat ===FILE:path=== ... ===END=== for each file. Nothing outside these blocks."""


def _parse_separator(text: str, framework: str) -> Dict[str, Any]:
    """Parse ===FILE:path=== ... ===END=== blocks — no JSON escaping issues.
    ===END=== must appear at the start of a line to avoid false matches inside code."""
    files = []
    for m in re.finditer(r'===FILE:([^\n=]+)===\s*(.*?)^===END===', text, re.DOTALL | re.MULTILINE):
        path = m.group(1).strip()
        code = m.group(2).strip()
        if path and code:
            files.append({"file_name": path, "code": code, "explanation": ""})
    if files:
        return {"framework": framework, "generated_files": files}
    # Fallback: try JSON if no separator blocks found
    return _parse_json_fallback(text, framework)


def _parse_json_fallback(text: str, framework: str) -> Dict[str, Any]:
    """Last-resort JSON parse — used only when separator format not found."""
    data = _parse_json(text)
    if "generated_files" not in data:
        data["generated_files"] = []
    data.setdefault("framework", framework)
    return data


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
    # Truncated — close open structures at last complete object
    partial = text[start:]
    last_brace = partial.rfind('}')
    if last_brace != -1:
        partial = partial[:last_brace+1]
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


_BATCH_SIZE = 5


def _slim_tc(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields Agent 3 needs — reduces input tokens."""
    slim = {
        "test_case_id":  tc.get("test_case_id", ""),
        "name":          tc.get("name", ""),
        "scenario_id":   tc.get("scenario_id", ""),
        "technique":     tc.get("technique", ""),
        "preconditions": tc.get("preconditions", ""),
        "steps":         tc.get("steps", []),
        "test_data":     tc.get("test_data", {}),
        "expected_result": tc.get("expected_result", ""),
        "priority":      tc.get("priority", "medium"),
    }
    # Include tags for context (security, boundary, etc.)
    if tc.get("tags"):
        slim["tags"] = tc["tags"]
    return slim


_DEFAULT_MODEL = "llama-3.1-8b-instant"


class AutomationCodeWriter:
    def __init__(self, model: str = None):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Export it before starting the server: export GROQ_API_KEY=<your-key>"
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("GROQ_MODEL", _DEFAULT_MODEL)

    def generate(
        self,
        test_cases: List[Dict[str, Any]],
        framework: str = "pytest",
        requirement_text: Optional[str] = None,
        payload_templates: Optional[List[Dict[str, Any]]] = None,
        test_data_matrix: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        all_files = []
        for i in range(0, max(len(test_cases), 1), _BATCH_SIZE):
            batch = test_cases[i:i + _BATCH_SIZE]
            result = self._generate_batch(batch, framework, requirement_text, payload_templates, test_data_matrix)
            all_files.extend(result.get("generated_files", []))
        return {"framework": framework, "generated_files": all_files}

    def _generate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        framework: str,
        requirement_text: Optional[str],
        payload_templates: Optional[List[Dict[str, Any]]],
        test_data_matrix: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        slim = [_slim_tc(tc) for tc in test_cases]
        tc_ids = {tc.get("test_case_id") for tc in slim}

        parts = []
        if requirement_text:
            parts.append(f"Requirement context:\n{requirement_text[:700]}")

        parts.append(
            f"Generate complete {framework} automation test files for these test cases:\n\n"
            f"{json.dumps(slim, ensure_ascii=False, indent=2)}"
        )

        if payload_templates:
            batch_tpl = [pt for pt in payload_templates if pt.get("test_case_id") in tc_ids]
            if batch_tpl:
                parts.append(f"Payload templates (use these for request bodies):\n{json.dumps(batch_tpl, ensure_ascii=False, indent=2)}")

        if test_data_matrix:
            batch_matrix = [row for row in test_data_matrix if row.get("test_case_id") in tc_ids]
            if batch_matrix:
                parts.append(f"Test data matrix:\n{json.dumps(batch_matrix, ensure_ascii=False, indent=2)}")

        user_msg = "\n\n".join(parts)

        def _once(msg=user_msg):
            completion = self.client.chat.completions.create(
                model=self.model, max_tokens=4000, temperature=0, seed=42,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": msg}],
            )
            text = completion.choices[0].message.content or ""
            if not text.strip():
                raise ValueError(f"Empty response (finish_reason={completion.choices[0].finish_reason})")
            return _parse_separator(text, framework)

        try:
            return call_with_backoff(_once, label="Agent3")
        except BadRequestError as e:
            if "json_validate_failed" in str(e):
                retry_msg = "Output ONLY the ===FILE:=== separator blocks, nothing else.\n\n" + user_msg
                return call_with_backoff(
                    lambda: _parse_separator(
                        self.client.chat.completions.create(
                            model=self.model, max_tokens=4000, temperature=0, seed=42,
                            messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": retry_msg}],
                        ).choices[0].message.content or "",
                        framework,
                    ),
                    label="Agent3-retry",
                )
            raise RuntimeError(f"Agent3 bad request: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Agent3 generation failed: {e}") from e
