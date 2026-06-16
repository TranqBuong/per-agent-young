"""Tests for parse_openapi — no LLM calls needed (pure parsing logic)."""
import json
import pytest

from backend.app.services.content_extractor import parse_openapi

# ── Fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_YAML = """
openapi: 3.0.0
info:
  title: Payment API
  version: 2.1.0
paths:
  /payments:
    post:
      summary: Create payment
      responses:
        "201":
          description: Created
"""

FULL_YAML = """
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: Manages user accounts
servers:
  - url: https://api.example.com/v1
paths:
  /users:
    get:
      summary: List users
      tags: [users]
      parameters:
        - name: page
          in: query
          required: false
          schema:
            type: integer
          description: Page number
        - name: Authorization
          in: header
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Success
        "401":
          description: Unauthorized
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [email, password]
              properties:
                email:
                  type: string
                  format: email
                  description: User email address
                password:
                  type: string
                role:
                  type: string
                  enum: [admin, user, guest]
      responses:
        "201":
          description: User created
        "400":
          description: Validation error
        "409":
          description: Email already exists
      security:
        - BearerAuth: []
  /users/{id}:
    delete:
      summary: Delete user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        "204":
          description: Deleted
"""

SWAGGER_V2 = """
swagger: "2.0"
info:
  title: Legacy API
  version: "1"
paths:
  /items:
    get:
      summary: Get items
      responses:
        200:
          description: OK
"""


def _parse(spec: str) -> str:
    return parse_openapi(spec)


# ── Basic parsing ─────────────────────────────────────────────────────────────

class TestParseOpenapiBasic:
    def test_parses_yaml(self):
        result = _parse(MINIMAL_YAML)
        assert "Payment API" in result

    def test_parses_json(self):
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "JSON API", "version": "1.0"},
            "paths": {"/ping": {"get": {"summary": "Ping", "responses": {"200": {"description": "OK"}}}}}
        })
        result = _parse(spec)
        assert "JSON API" in result
        assert "/ping" in result

    def test_includes_title_and_version(self):
        result = _parse(FULL_YAML)
        assert "User API" in result
        assert "1.0.0" in result

    def test_includes_description(self):
        result = _parse(FULL_YAML)
        assert "Manages user accounts" in result

    def test_includes_base_url(self):
        result = _parse(FULL_YAML)
        assert "https://api.example.com/v1" in result

    def test_swagger_v2_accepted(self):
        result = _parse(SWAGGER_V2)
        assert "Legacy API" in result
        assert "/items" in result


# ── Endpoints ─────────────────────────────────────────────────────────────────

class TestParseOpenapiEndpoints:
    def test_http_method_uppercased(self):
        result = _parse(FULL_YAML)
        assert "GET /users" in result
        assert "POST /users" in result
        assert "DELETE /users/{id}" in result

    def test_summary_included(self):
        result = _parse(FULL_YAML)
        assert "List users" in result
        assert "Create user" in result

    def test_tags_included(self):
        result = _parse(FULL_YAML)
        assert "users" in result

    def test_multiple_endpoints_all_present(self):
        result = _parse(FULL_YAML)
        assert "/users" in result
        assert "/users/{id}" in result


# ── Parameters ────────────────────────────────────────────────────────────────

class TestParseOpenapiParameters:
    def test_query_param_listed(self):
        result = _parse(FULL_YAML)
        assert "page" in result

    def test_param_location_shown(self):
        result = _parse(FULL_YAML)
        assert "query" in result
        assert "header" in result
        assert "path" in result

    def test_required_param_marked(self):
        result = _parse(FULL_YAML)
        assert "(required)" in result

    def test_optional_param_marked(self):
        result = _parse(FULL_YAML)
        assert "(optional)" in result

    def test_param_type_shown(self):
        result = _parse(FULL_YAML)
        assert "integer" in result or "string" in result

    def test_param_description_included(self):
        result = _parse(FULL_YAML)
        assert "Page number" in result


# ── Request body ──────────────────────────────────────────────────────────────

class TestParseOpenapiRequestBody:
    def test_request_body_section_present(self):
        result = _parse(FULL_YAML)
        assert "Request body" in result

    def test_field_names_listed(self):
        result = _parse(FULL_YAML)
        assert "email" in result
        assert "password" in result

    def test_field_type_shown(self):
        result = _parse(FULL_YAML)
        assert "string" in result

    def test_required_fields_marked(self):
        result = _parse(FULL_YAML)
        # email and password are required
        assert "(required)" in result

    def test_optional_fields_marked(self):
        result = _parse(FULL_YAML)
        # role is optional
        assert "(optional)" in result

    def test_enum_values_shown(self):
        result = _parse(FULL_YAML)
        assert "admin" in result
        assert "user" in result
        assert "guest" in result

    def test_format_shown(self):
        result = _parse(FULL_YAML)
        assert "email" in result  # format: email shown

    def test_field_description_included(self):
        result = _parse(FULL_YAML)
        assert "User email address" in result


# ── Responses ─────────────────────────────────────────────────────────────────

class TestParseOpenapiResponses:
    def test_response_codes_listed(self):
        result = _parse(FULL_YAML)
        assert "201" in result
        assert "400" in result
        assert "409" in result

    def test_response_description_shown(self):
        result = _parse(FULL_YAML)
        assert "User created" in result
        assert "Validation error" in result
        assert "Email already exists" in result

    def test_minimal_responses(self):
        result = _parse(MINIMAL_YAML)
        assert "201" in result
        assert "Created" in result


# ── Security ──────────────────────────────────────────────────────────────────

class TestParseOpenapiSecurity:
    def test_security_scheme_shown(self):
        result = _parse(FULL_YAML)
        assert "BearerAuth" in result


# ── $ref resolution ───────────────────────────────────────────────────────────

class TestParseOpenapiRefResolution:
    def test_resolves_component_ref(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Ref API", "version": "1"},
            "components": {
                "schemas": {
                    "CreatePaymentRequest": {
                        "type": "object",
                        "required": ["amount"],
                        "properties": {
                            "amount": {"type": "number", "description": "Amount in VND"},
                            "note":   {"type": "string"},
                        }
                    }
                }
            },
            "paths": {
                "/payments": {
                    "post": {
                        "summary": "Pay",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CreatePaymentRequest"}
                                }
                            }
                        },
                        "responses": {"200": {"description": "OK"}}
                    }
                }
            }
        }
        result = _parse(json.dumps(spec))
        assert "amount" in result
        assert "Amount in VND" in result
        assert "note" in result


# ── Error cases ───────────────────────────────────────────────────────────────

class TestParseOpenapiErrors:
    def test_raises_on_unparseable_content(self):
        with pytest.raises(ValueError):
            parse_openapi("this is not yaml or json at all }{[")

    def test_raises_on_non_openapi_json(self):
        with pytest.raises(ValueError):
            parse_openapi('{"name": "foo", "value": 42}')

    def test_raises_on_empty_object(self):
        with pytest.raises(ValueError):
            parse_openapi("{}")

    def test_raises_if_no_paths_key(self):
        with pytest.raises(ValueError):
            parse_openapi('{"info": {"title": "x"}}')

    def test_plain_text_raises(self):
        with pytest.raises(ValueError):
            parse_openapi("POST /login\n- email: required")

    def test_returns_string(self):
        result = _parse(MINIMAL_YAML)
        assert isinstance(result, str)
        assert len(result) > 0
