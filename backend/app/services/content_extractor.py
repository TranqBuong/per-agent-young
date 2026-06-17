import json
import os
import re
import yaml
from openai import OpenAI
from backend.app.services.groq_retry import call_with_backoff

_client = None

_SYSTEM_PROMPT = """You are a requirements extraction specialist.

Given raw text from a document or webpage, extract ONLY:
- Software / API requirements and specifications
- Functional requirements and user stories
- Business rules and constraints
- Technical specs, data formats, field validations
- Acceptance criteria and test conditions
- Endpoint definitions (method, path, request/response)

REMOVE completely:
- Navigation menus, sidebars, headers, footers
- Advertisements, marketing copy, pricing pages
- Boilerplate, copyright notices, legal disclaimers
- HTML artifacts, CSS/JS code, encoding artifacts
- Timestamps, breadcrumbs, unrelated prose
- Repeated or duplicate content

Output rules:
- Return ONLY the extracted requirement text, no explanation or preamble
- Preserve original wording and structure of requirements
- Keep bullet points, numbered lists, and table-like data as-is
- If the input is already clean requirement text, return it unchanged
- If no clear requirements are found, return the most technically relevant content"""


def _strip_html(text: str) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-z]{2,6};', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GREENNODE_AIP_KEY")
        if not api_key:
            raise RuntimeError(
                "GREENNODE_AIP_KEY environment variable is not set. "
                "Export it before starting the server: export GREENNODE_AIP_KEY=<your-key>"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("GREENNODE_AIP_BASE_URL",
                                    "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"),
        )
    return _client


def _fmt_schema(schema: dict, indent: int = 0) -> str:
    """Recursively format a JSON Schema object into readable text."""
    if not isinstance(schema, dict):
        return str(schema)
    lines = []
    pad = "  " * indent
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, prop in props.items():
        prop_type = prop.get("type", prop.get("$ref", "object"))
        if isinstance(prop_type, list):
            prop_type = "/".join(prop_type)
        req_marker = " (required)" if name in required else " (optional)"
        desc = prop.get("description", "")
        enum = prop.get("enum")
        fmt = prop.get("format", "")
        extra = ""
        if enum:
            extra = f" [enum: {', '.join(str(v) for v in enum)}]"
        elif fmt:
            extra = f" [{fmt}]"
        desc_part = f" — {desc}" if desc else ""
        lines.append(f"{pad}- {name}: {prop_type}{req_marker}{extra}{desc_part}")
        if prop.get("properties"):
            lines.append(_fmt_schema(prop, indent + 1))
    return "\n".join(lines)


def parse_openapi(raw: str) -> str:
    """Parse OpenAPI/Swagger JSON or YAML and return structured requirement text."""
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError:
        try:
            spec = yaml.safe_load(raw)
        except Exception as exc:
            raise ValueError(f"Could not parse as JSON or YAML: {exc}") from exc

    if not isinstance(spec, dict):
        raise ValueError("Parsed content is not an object")

    # Detect OpenAPI/Swagger
    if "openapi" not in spec and "swagger" not in spec and "paths" not in spec:
        raise ValueError("Not a valid OpenAPI/Swagger spec (missing openapi/swagger/paths key)")

    lines = []

    # Title & description
    info = spec.get("info", {})
    title = info.get("title", "API")
    version = info.get("version", "")
    description = info.get("description", "")
    lines.append(f"# {title}" + (f" (v{version})" if version else ""))
    if description:
        lines.append(description.strip())
    lines.append("")

    # Servers / base URL
    servers = spec.get("servers", [])
    if servers:
        base = servers[0].get("url", "")
        if base:
            lines.append(f"Base URL: {base}")
            lines.append("")

    # Paths / endpoints
    paths = spec.get("paths", {})
    components = spec.get("components", spec.get("definitions", {}))
    schemas = components.get("schemas", components) if isinstance(components, dict) else {}

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            op = path_item.get(method)
            if not op:
                continue
            summary = op.get("summary", "")
            desc_op = op.get("description", "")
            op_id = op.get("operationId", "")
            tags = op.get("tags", [])

            lines.append(f"## {method.upper()} {path}")
            if summary:
                lines.append(f"Summary: {summary}")
            if desc_op:
                lines.append(desc_op.strip())
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")

            # Parameters
            params = op.get("parameters", []) + path_item.get("parameters", [])
            if params:
                lines.append("Parameters:")
                for p in params:
                    if not isinstance(p, dict):
                        continue
                    pname = p.get("name", "?")
                    pin = p.get("in", "query")
                    preq = " (required)" if p.get("required") else " (optional)"
                    pschema = p.get("schema", {})
                    ptype = pschema.get("type", "string") if isinstance(pschema, dict) else "string"
                    pdesc = p.get("description", "")
                    desc_part = f" — {pdesc}" if pdesc else ""
                    lines.append(f"  - {pname} [{pin}]: {ptype}{preq}{desc_part}")

            # Request body
            req_body = op.get("requestBody", {})
            if req_body:
                rb_desc = req_body.get("description", "")
                rb_req = " (required)" if req_body.get("required") else ""
                lines.append(f"Request body{rb_req}:" + (f" {rb_desc}" if rb_desc else ""))
                content = req_body.get("content", {})
                for media_type, media_obj in content.items():
                    if not isinstance(media_obj, dict):
                        continue
                    rb_schema = media_obj.get("schema", {})
                    if "$ref" in rb_schema:
                        ref_name = rb_schema["$ref"].split("/")[-1]
                        rb_schema = schemas.get(ref_name, rb_schema)
                    schema_text = _fmt_schema(rb_schema)
                    if schema_text:
                        lines.append(f"  [{media_type}]")
                        lines.append(schema_text)

            # Responses
            responses = op.get("responses", {})
            if responses:
                lines.append("Responses:")
                for status_code, resp in responses.items():
                    if not isinstance(resp, dict):
                        continue
                    rdesc = resp.get("description", "")
                    lines.append(f"  - {status_code}: {rdesc}")
                    resp_content = resp.get("content", {})
                    for media_type, media_obj in resp_content.items():
                        if not isinstance(media_obj, dict):
                            continue
                        rs = media_obj.get("schema", {})
                        if "$ref" in rs:
                            ref_name = rs["$ref"].split("/")[-1]
                            rs = schemas.get(ref_name, rs)
                        rs_text = _fmt_schema(rs)
                        if rs_text:
                            lines.append(f"    [{media_type}]")
                            lines.append(rs_text)

            # Security
            security = op.get("security", spec.get("security", []))
            if security:
                sec_names = [list(s.keys())[0] for s in security if s]
                lines.append(f"Auth: {', '.join(sec_names)}")

            lines.append("")

    return "\n".join(lines).strip()


def extract_requirements(raw_text: str, source_hint: str = "") -> str:
    """Strip noise and extract relevant requirements using LLM."""
    cleaned = _strip_html(raw_text)
    input_text = cleaned[:5000]

    client = _get_client()
    source_line = f"Source: {source_hint}\n\n" if source_hint else ""

    def _once():
        completion = client.chat.completions.create(
            model=os.environ.get("GREENNODE_MODEL_LIGHT", "deepseek/deepseek-v4-flash"),
            max_tokens=2000,
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"{source_line}Extract requirements from:\n\n{input_text}"},
            ],
        )
        return (completion.choices[0].message.content or "").strip()

    try:
        result = call_with_backoff(_once, label="ContentExtractor")
        return result if result else cleaned[:8000]
    except Exception:
        return cleaned[:8000]
