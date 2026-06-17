"""Tests for pure utility functions — no LLM/network calls."""
import pytest

from backend.app.agents.requirement_analyzer import _sanitize_text
from backend.app.services.content_extractor import _strip_html, _fmt_schema


# ── _sanitize_text ────────────────────────────────────────────────────────────

class TestSanitizeText:
    def test_replaces_double_quotes_with_single(self):
        assert _sanitize_text('say "hello"') == "say 'hello'"

    def test_replaces_backslash_with_forward_slash(self):
        assert _sanitize_text('path\\to\\file') == 'path/to/file'

    def test_both_replacements_applied(self):
        text = 'field \\"value\\"'
        result = _sanitize_text(text)
        assert '"' not in result
        assert '\\' not in result

    def test_empty_string_unchanged(self):
        assert _sanitize_text('') == ''

    def test_clean_text_unchanged(self):
        text = 'POST /login endpoint requires email and password'
        assert _sanitize_text(text) == text

    def test_mixed_content(self):
        result = _sanitize_text('Use "Bearer token" for \\Authorization')
        assert result == "Use 'Bearer token' for /Authorization"

    def test_multiple_quotes_all_replaced(self):
        result = _sanitize_text('"a" and "b" and "c"')
        assert '"' not in result
        assert result == "'a' and 'b' and 'c'"

    def test_multiple_backslashes_all_replaced(self):
        result = _sanitize_text('a\\b\\c\\d')
        assert '\\' not in result
        assert result == 'a/b/c/d'

    def test_unicode_text_unchanged(self):
        text = 'Người dùng đăng nhập bằng email'
        assert _sanitize_text(text) == text

    def test_quote_inside_unicode_text(self):
        result = _sanitize_text('Xác thực "token" cho người dùng')
        assert result == "Xác thực 'token' cho người dùng"


# ── _strip_html ───────────────────────────────────────────────────────────────

class TestStripHtml:
    def test_removes_basic_tags(self):
        result = _strip_html("<p>Hello World</p>")
        assert "<p>" not in result
        assert "Hello World" in result

    def test_removes_nested_tags(self):
        result = _strip_html("<div><span><b>text</b></span></div>")
        assert "text" in result
        assert "<" not in result

    def test_removes_script_block_and_content(self):
        html = "<p>visible</p><script>alert('xss')</script>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "xss" not in result
        assert "visible" in result

    def test_removes_style_block_and_content(self):
        html = "<style>body { color: red; font-size: 12px; }</style><p>text</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "font-size" not in result
        assert "text" in result

    def test_replaces_nbsp_entity(self):
        result = _strip_html("Hello&nbsp;World")
        assert "&nbsp;" not in result

    def test_removes_html_entities(self):
        result = _strip_html("&amp;test&lt;value&gt;")
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_collapses_multiple_spaces(self):
        result = _strip_html("<p>a    b</p>")
        assert "  " not in result

    def test_collapses_many_newlines_to_max_two(self):
        result = _strip_html("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_returns_stripped_string(self):
        result = _strip_html("   <b>text</b>   ")
        assert result == result.strip()

    def test_plain_text_unchanged_except_whitespace(self):
        result = _strip_html("just plain text")
        assert "just plain text" in result

    def test_self_closing_tags_removed(self):
        result = _strip_html("<br/>text<img src='x'/>more")
        assert "text" in result
        assert "more" in result
        assert "<" not in result

    def test_multiline_script_removed(self):
        html = "<p>keep</p><script type='text/javascript'>\nvar x = 1;\nalert(x);\n</script>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "keep" in result


# ── _fmt_schema ───────────────────────────────────────────────────────────────

class TestFmtSchema:
    def test_basic_required_property(self):
        schema = {"properties": {"email": {"type": "string"}}, "required": ["email"]}
        result = _fmt_schema(schema)
        assert "email" in result
        assert "string" in result
        assert "(required)" in result

    def test_optional_property_marked(self):
        schema = {"properties": {"note": {"type": "string"}}, "required": []}
        result = _fmt_schema(schema)
        assert "(optional)" in result

    def test_enum_values_listed(self):
        schema = {"properties": {"role": {"type": "string", "enum": ["admin", "user", "guest"]}}}
        result = _fmt_schema(schema)
        assert "admin" in result
        assert "user" in result
        assert "guest" in result

    def test_format_shown(self):
        schema = {"properties": {"created_at": {"type": "string", "format": "date-time"}}}
        result = _fmt_schema(schema)
        assert "date-time" in result

    def test_description_included(self):
        schema = {"properties": {"amount": {"type": "number", "description": "Payment amount in VND"}}}
        result = _fmt_schema(schema)
        assert "Payment amount in VND" in result

    def test_empty_schema_returns_empty_string(self):
        result = _fmt_schema({})
        assert result == ""

    def test_schema_without_properties_returns_empty(self):
        result = _fmt_schema({"type": "object", "required": []})
        assert result == ""

    def test_non_dict_returns_string_representation(self):
        assert _fmt_schema("not a dict") == "not a dict"
        assert _fmt_schema(42) == "42"

    def test_multiple_properties_all_shown(self):
        schema = {
            "properties": {
                "email": {"type": "string"},
                "password": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["email", "password"],
        }
        result = _fmt_schema(schema)
        assert "email" in result
        assert "password" in result
        assert "age" in result

    def test_required_vs_optional_distinction(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        result = _fmt_schema(schema)
        lines = result.splitlines()
        name_line = next(l for l in lines if "name" in l and "nick" not in l)
        nick_line = next(l for l in lines if "nickname" in l)
        assert "(required)" in name_line
        assert "(optional)" in nick_line

    def test_nested_properties_shown(self):
        schema = {
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                }
            }
        }
        result = _fmt_schema(schema)
        assert "address" in result
        assert "street" in result
        assert "city" in result

    def test_nested_indentation_increases(self):
        schema = {
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                }
            }
        }
        result = _fmt_schema(schema)
        lines = result.splitlines()
        outer_line = next(l for l in lines if "outer" in l)
        inner_line = next(l for l in lines if "inner" in l)
        # Inner should be indented more than outer
        assert len(inner_line) - len(inner_line.lstrip()) > len(outer_line) - len(outer_line.lstrip())

    def test_list_type_joined(self):
        schema = {"properties": {"value": {"type": ["string", "null"]}}}
        result = _fmt_schema(schema)
        assert "string" in result
        assert "null" in result
