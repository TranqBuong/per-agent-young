import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path as _Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import ResponseValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
from pathlib import Path
from backend.app.workflows.mvp_workflow import MVPWorkflow
from backend.app.workflows.test_case_payload_workflow import TestCasePayloadWorkflow
from backend.app.workflows.automation_code_workflow import AutomationCodeWorkflow
from backend.app.workflows.full_pipeline_workflow import FullPipelineWorkflow
from pydantic import BaseModel
from backend.app.schemas.agent_schemas import (
    RequirementInput,
    RequirementPreviewResult,
    GenerateTestCasesInput,
    GenerateAutomationCodeInput,
)
from backend.app.services.cache_service import CacheService
from backend.app.services.content_extractor import extract_requirements, parse_openapi

app = FastAPI(
    title="Multi-Agent Automation Engineer",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


_logger = logging.getLogger(__name__)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Endpoint '{request.url.path}' not found. Visit /docs for the list of available endpoints."},
        )
    if exc.status_code == 405:
        return JSONResponse(
            status_code=405,
            content={"detail": f"Method '{request.method}' is not allowed on '{request.url.path}'. Check /docs for the correct HTTP method."},
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(ResponseValidationError)
async def response_validation_error_handler(request: Request, exc: ResponseValidationError):
    _logger.error("Response validation error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Response serialization error."})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    _logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred. Please try again."})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin is incompatible with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parent.parent / "frontend"


@app.get("/")
def serve_frontend():
    return FileResponse(_FRONTEND / "index.html")
workflow = MVPWorkflow()
test_case_workflow = TestCasePayloadWorkflow()
automation_code_workflow = AutomationCodeWorkflow()
full_pipeline_workflow = FullPipelineWorkflow()
cache = CacheService()


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Multi-Agent Automation Engineer backend is running"}


def _inject_base_url(code: str, base_url: str) -> str:
    replacement = f'BASE_URL = "{base_url}"'
    new_code, n = re.subn(r'BASE_URL\s*=\s*["\'].*?["\']', replacement, code)
    if n == 0:
        new_code = replacement + "\n" + code
    return new_code


def _parse_pytest_output(output: str) -> dict:
    passed = failed = errors = 0
    for line in output.splitlines():
        m = re.search(r'(\d+)\s+passed', line, re.IGNORECASE)
        if m:
            passed = max(passed, int(m.group(1)))
        m = re.search(r'(\d+)\s+failed', line, re.IGNORECASE)
        if m:
            failed = max(failed, int(m.group(1)))
        m = re.search(r'(\d+)\s+error', line, re.IGNORECASE)
        if m:
            errors = max(errors, int(m.group(1)))
    test_results = [
        line for line in output.splitlines()
        if line.startswith(("PASSED", "FAILED", "ERROR"))
    ]
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": passed + failed + errors,
        "test_results": test_results,
    }


def _timed(data: dict, elapsed: float, cached: bool = False) -> JSONResponse:
    data["_meta"] = {"elapsed_ms": round(elapsed * 1000), "cached": cached}
    return JSONResponse(content=data)


_MAX_FILE_CHARS = 8000
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/parse-file")
async def parse_file(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB allowed.")

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(pages).strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")

    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DOCX parse error: {e}")

    elif name.endswith(".txt") or name.endswith(".md"):
        try:
            content = data.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Text decode error: {e}")

    elif name.endswith(".yaml") or name.endswith(".yml") or name.endswith(".json"):
        try:
            raw = data.decode("utf-8", errors="replace")
            content = parse_openapi(raw)
            # Already structured — return directly, no LLM extraction needed
            return {
                "content": content[:_MAX_FILE_CHARS],
                "truncated": len(content) > _MAX_FILE_CHARS,
                "original_length": len(content),
            }
        except ValueError:
            # Not a valid OpenAPI spec — fall through to plain text path
            content = data.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OpenAPI parse error: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    if not content.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")

    original_length = len(content)
    extracted = extract_requirements(content, source_hint=file.filename or "")
    truncated = len(extracted) > _MAX_FILE_CHARS
    return {
        "content": extracted[:_MAX_FILE_CHARS],
        "truncated": truncated,
        "original_length": original_length,
    }


_MAX_URL_CHARS = 8000


@app.get("/fetch-url")
def fetch_url(url: str = Query(..., description="URL to fetch text content from")):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; QA-Bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            encoding = resp.headers.get_content_charset("utf-8") or "utf-8"
            content = raw.decode(encoding, errors="replace")
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=400, detail=f"URL error: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    original_length = len(content)
    extracted = extract_requirements(content, source_hint=url)
    truncated = len(extracted) > _MAX_URL_CHARS
    return {
        "content": extracted[:_MAX_URL_CHARS],
        "truncated": truncated,
        "original_length": original_length,
    }


@app.post("/preview-requirements")
def preview_requirements(payload: RequirementInput):
    t0 = time.time()
    cached = cache.get("preview", payload.text)
    if cached:
        return _timed(cached, time.time() - t0, cached=True)
    try:
        raw = workflow.analyzer.preview(payload.text)
        # Strip internal debug fields before schema validation
        raw.pop("_input_length", None)
        raw.pop("quality_checks", None)
        validated = RequirementPreviewResult(**raw)
        data = validated.model_dump()
        cache.set("preview", payload.text, data)
        return _timed(data, time.time() - t0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 1 preview failed: {e}")


@app.post("/analyze-requirements")
def analyze_requirements(payload: RequirementInput):
    t0 = time.time()
    cached = cache.get("analyze", payload.text)
    if cached:
        return _timed(cached, time.time() - t0, cached=True)
    try:
        result = workflow.run(payload.text)
        data = result if isinstance(result, dict) else result.model_dump()
        cache.set("analyze", payload.text, data)
        return _timed(data, time.time() - t0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 1 analyze failed: {e}")


@app.post("/generate-test-cases")
def generate_test_cases(payload: GenerateTestCasesInput):
    t0 = time.time()
    req_text = payload.requirement_text or ""
    cached = cache.get("testcases", req_text) if req_text else None
    if cached:
        return _timed(cached, time.time() - t0, cached=True)
    try:
        result = test_case_workflow.run(
            payload.scenarios,
            requirement_text=req_text,
            overview=payload.overview,
        )
        data = result if isinstance(result, dict) else result.model_dump()
        if req_text:
            cache.set("testcases", req_text, data)
        return _timed(data, time.time() - t0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 2 failed: {e}")


@app.post("/generate-automation-code")
def generate_automation_code(payload: GenerateAutomationCodeInput):
    t0 = time.time()
    req_text = payload.requirement_text or ""
    framework = payload.framework or "pytest"
    serialized_test_cases = [
        tc.model_dump() if hasattr(tc, "model_dump") else tc
        for tc in payload.test_cases
    ]
    tc_hash = hashlib.md5(
        json.dumps(serialized_test_cases, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:8]
    cache_key = f"{req_text}|{framework}|{tc_hash}" if req_text else None
    if cache_key:
        cached = cache.get("autocode", cache_key)
        if cached:
            return _timed(cached, time.time() - t0, cached=True)
    try:
        result = automation_code_workflow.run(
            payload.test_cases,
            framework=framework,
            requirement_text=req_text,
            payload_templates=payload.payload_templates,
            test_data_matrix=payload.test_data_matrix,
        )
        data = result if isinstance(result, dict) else result.model_dump()
        if cache_key:
            cache.set("autocode", cache_key, data)
        return _timed(data, time.time() - t0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 3 failed: {e}")



_RUNNABLE_FRAMEWORKS = {"pytest"}
_MAX_RUN_OUTPUT = 8000


class _RunTestFile(BaseModel):
    file_name: str
    code: str


class RunTestsInput(BaseModel):
    files: List[_RunTestFile]
    framework: str = "pytest"
    base_url: Optional[str] = ""


@app.post("/run-tests")
def run_tests(payload: RunTestsInput):
    if not payload.files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    base_url = (payload.base_url or "").strip()
    if base_url and not base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="base_url must start with http:// or https://")

    if payload.framework not in _RUNNABLE_FRAMEWORKS:
        return {
            "supported": False,
            "message": f"Framework '{payload.framework}' cannot be executed server-side. Only pytest is supported.",
            "passed": 0, "failed": 0, "errors": 0, "total": 0,
            "output": "", "test_results": [], "return_code": -1,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in payload.files:
            code = _inject_base_url(f.code, base_url) if base_url else f.code
            (_Path(tmpdir) / f.file_name).write_text(code, encoding="utf-8")

        proc = subprocess.run(
            ["python", "-m", "pytest", "-v", tmpdir],
            capture_output=True, text=True, timeout=120,
        )
        raw_output = proc.stdout + proc.stderr
        output = raw_output[:_MAX_RUN_OUTPUT]
        parsed = _parse_pytest_output(raw_output)

    return {
        "supported": True,
        "message": "Tests executed",
        **parsed,
        "output": output,
        "return_code": proc.returncode,
    }


@app.post("/run-full-pipeline")
def run_full_pipeline(payload: RequirementInput):
    try:
        result = full_pipeline_workflow.run(payload)
        return {
            "message": "Full pipeline completed",
            "artifacts": {
                "json": str(full_pipeline_workflow.output_dir / "full_pipeline_output.json"),
                "markdown": str(full_pipeline_workflow.output_dir / "full_pipeline_output.md"),
            },
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full pipeline failed: {e}")
