import ast
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from openai import OpenAI, BadRequestError
from backend.app.services.groq_retry import call_with_backoff

_logger = logging.getLogger(__name__)

_FRAMEWORK_GUIDES = {
    "pytest":     "use `requests` library, `conftest.py` with fixtures (base_url, auth headers), clear assertions.",
    "playwright": "use `playwright-python` sync API, `pytest-playwright` fixtures.",
    "k6":         "use k6 JavaScript API (`import http from 'k6/http'`), `check()` assertions, export `default function`.",
    "selenium":   "use `selenium` WebDriver with `pytest`, `webdriver.Chrome()` with `Options`, explicit `WebDriverWait`, `By` selectors.",
    "postman":    "generate a valid Postman Collection v2.1 JSON (`info`, `item[]` array). Each TC → one `item` with `request` and `event` (test scripts using `pm.test`/`pm.response.to.have.status`). Output a single `.json` collection file.",
}

_SYSTEM_PROMPT_TEMPLATE = """You are an expert test automation engineer.
Your ONLY task: generate executable {framework} test code for each test case provided.

## Framework: {framework}
{guide}

## Input fields (each test case from Agent 2)
test_case_id | name | steps | test_data | expected_result | technique | scenario_id

## Rules
1. ONE test function per test case — never merge, split, or add extras.
2. Use exact values from `test_data` — never invent or substitute values.
3. Follow `steps` in order — each step maps to a code statement.
4. Assert exactly what `expected_result` states.
5. Every function must be complete and runnable — no `...`, `pass`, or TODO stubs.
6. Use `os.environ.get("BASE_URL", "http://localhost:8000")` for the base URL.
7. Function naming: `def test_{{test_case_id}}_{{snake_case_name}}():`.
8. Apply `technique` to assertions:
   BVA → exact boundary values | EG/EP → HTTP status codes or error messages | DT → decision-table combinations | UC → end-to-end flow outcome.

## Output Format — REQUIRED
===FILE:tests/test_{{test_case_id}}.py===
<complete runnable code>
===END===

One ===FILE=== block per test case, named after test_case_id.
Output ONLY ===FILE=== blocks — no markdown, no explanation, nothing outside them.
Do NOT generate conftest.py, fixtures files, or any helper files — test functions only."""


def _system_prompt(framework: str) -> str:
    guide = _FRAMEWORK_GUIDES.get(framework, _FRAMEWORK_GUIDES["pytest"])
    return _SYSTEM_PROMPT_TEMPLATE.format(framework=framework, guide=guide)


def _parse_separator(text: str, framework: str) -> Dict[str, Any]:
    """Parse ===FILE:path=== ... ===END=== blocks — no JSON escaping issues."""
    files = []
    for m in re.finditer(r'===FILE:([^\n=]+)===\s*(.*?)^===END===', text, re.DOTALL | re.MULTILINE):
        path = m.group(1).strip()
        code = m.group(2).strip()
        if path and code:
            files.append({"file_name": path, "code": code, "explanation": ""})
    if files:
        return {"framework": framework, "generated_files": files}

    # Truncated response: ===FILE:path=== present but ===END=== missing (cut by max_tokens)
    m = re.search(r'===FILE:([^\n=]+)===\s*([\s\S]+)', text)
    if m:
        path = m.group(1).strip()
        code = m.group(2).strip()
        if path and code:
            return {"framework": framework, "generated_files": [{"file_name": path, "code": code, "explanation": ""}]}

    # Last-resort JSON fallback — only if response actually looks like JSON
    stripped = text.strip()
    if stripped.startswith('{') or stripped.startswith('['):
        return _parse_json_fallback(text, framework)

    return {"framework": framework, "generated_files": []}


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


def _normalize_json_like(text: str) -> str:
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(
        r'(?P<prefix>[{\s,])(?P<key>[A-Za-z_][A-Za-z0-9_]*)' r'(?P<suffix>\s*:)',
        r'\g<prefix>"\g<key>"\g<suffix>',
        text,
    )
    return text


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = _extract_json(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            normalized = _normalize_json_like(text)
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                return ast.literal_eval(normalized)


_BATCH_SIZE = 1
_MAX_WORKERS = 10  # concurrent LLM calls for Agent 3


def _slim_tc(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields Agent 3 needs — reduces input tokens."""
    return {
        "test_case_id":    tc.get("test_case_id", ""),
        "scenario_id":     tc.get("scenario_id", ""),
        "technique":       tc.get("technique", ""),
        "name":            tc.get("name", ""),
        "steps":           tc.get("steps", []),
        "test_data":       tc.get("test_data", {}),
        "expected_result": tc.get("expected_result", ""),
        "preconditions":   tc.get("preconditions", ""),
    }


_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_AIP_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"


class AutomationCodeWriter:
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

    def generate(
        self,
        test_cases: List[Dict[str, Any]],
        framework: str = "pytest",
        requirement_text: Optional[str] = None,
        payload_templates: Optional[List[Dict[str, Any]]] = None,
        test_data_matrix: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not test_cases:
            _logger.warning("Agent3: no test_cases provided — returning empty result")
            return {"framework": framework, "generated_files": []}

        batches = [test_cases[i:i + _BATCH_SIZE] for i in range(0, len(test_cases), _BATCH_SIZE)]
        results_by_index: Dict[int, Dict] = {}

        with ThreadPoolExecutor(max_workers=min(len(batches), _MAX_WORKERS)) as executor:
            future_to_idx = {
                executor.submit(self._generate_batch, batch, framework): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_by_index[idx] = future.result()
                except Exception as exc:
                    _logger.error("Agent3 batch %d failed: %s", idx, exc)
                    results_by_index[idx] = {"generated_files": []}

        all_files = []
        for idx in sorted(results_by_index.keys()):
            all_files.extend(results_by_index[idx].get("generated_files", []))

        return {"framework": framework, "generated_files": all_files}

    def _generate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        framework: str,
    ) -> Dict[str, Any]:
        slim = [_slim_tc(tc) for tc in test_cases]

        user_msg = (
            f"Generate {framework} automation test files for these test cases:\n\n"
            f"{json.dumps(slim, ensure_ascii=False)}"
        )

        def _once(msg=user_msg):
            completion = self.client.chat.completions.create(
                model=self.model, max_tokens=3000, temperature=0,
                messages=[{"role": "system", "content": _system_prompt(framework)}, {"role": "user", "content": msg}],
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
                            model=self.model, max_tokens=3000, temperature=0,
                            messages=[{"role": "system", "content": _system_prompt(framework)}, {"role": "user", "content": retry_msg}],
                        ).choices[0].message.content or "",
                        framework,
                    ),
                    label="Agent3-retry",
                )
            raise RuntimeError(f"Agent3 bad request: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Agent3 generation failed: {e}") from e
