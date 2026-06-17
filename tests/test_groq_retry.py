"""Tests for groq_retry.call_with_backoff — no real network calls."""
import pytest
from unittest.mock import MagicMock, patch

from openai import RateLimitError, APIConnectionError, APIStatusError, BadRequestError

from backend.app.services.groq_retry import call_with_backoff, _MAX_RETRIES, _BASE_DELAY


def _rate_limit():
    r = MagicMock(); r.status_code = 429; r.headers = {}
    return RateLimitError("rate limited", response=r, body=None)


def _connection_error():
    return APIConnectionError.__new__(APIConnectionError)


def _bad_request():
    r = MagicMock(); r.status_code = 400; r.headers = {}
    return BadRequestError("json_validate_failed", response=r, body=None)


def _api_status(code=500):
    r = MagicMock(); r.status_code = code; r.headers = {}
    return APIStatusError("server error", response=r, body=None)


class TestSuccessPath:
    def test_returns_value_on_first_try(self):
        fn = MagicMock(return_value="ok")
        assert call_with_backoff(fn) == "ok"
        assert fn.call_count == 1

    def test_no_sleep_when_succeeds_immediately(self):
        fn = MagicMock(return_value=42)
        with patch("backend.app.services.groq_retry.time.sleep") as mock_sleep:
            call_with_backoff(fn)
        mock_sleep.assert_not_called()

    def test_label_parameter_ignored_on_success(self):
        fn = MagicMock(return_value={"key": "value"})
        result = call_with_backoff(fn, label="MyAgent")
        assert result == {"key": "value"}


class TestRateLimitRetry:
    def test_retries_twice_then_succeeds(self):
        err = _rate_limit()
        fn = MagicMock(side_effect=[err, err, "ok"])
        with patch("backend.app.services.groq_retry.time.sleep"):
            result = call_with_backoff(fn, "test")
        assert result == "ok"
        assert fn.call_count == 3

    def test_raises_runtime_error_after_max_retries(self):
        err = _rate_limit()
        fn = MagicMock(side_effect=[err] * _MAX_RETRIES)
        with patch("backend.app.services.groq_retry.time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                call_with_backoff(fn, "test")
        assert "rate limit" in str(exc_info.value).lower()
        assert fn.call_count == _MAX_RETRIES

    def test_sleep_called_with_exponential_backoff(self):
        err = _rate_limit()
        fn = MagicMock(side_effect=[err, err, "ok"])
        with patch("backend.app.services.groq_retry.time.sleep") as mock_sleep:
            call_with_backoff(fn)
        # attempt 0 → 1.5 * 2^0 = 1.5; attempt 1 → 1.5 * 2^1 = 3.0
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [_BASE_DELAY * (2 ** 0), _BASE_DELAY * (2 ** 1)]

    def test_no_sleep_on_final_rate_limit_attempt(self):
        """On the last retry (attempt == _MAX_RETRIES-1) we raise, not sleep."""
        err = _rate_limit()
        fn = MagicMock(side_effect=[err] * _MAX_RETRIES)
        with patch("backend.app.services.groq_retry.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                call_with_backoff(fn)
        # Only _MAX_RETRIES - 1 sleeps (no sleep before raise on last attempt)
        assert mock_sleep.call_count == _MAX_RETRIES - 1


class TestConnectionErrorRetry:
    def test_retries_on_connection_error_then_succeeds(self):
        err = _connection_error()
        fn = MagicMock(side_effect=[err, "ok"])
        with patch("backend.app.services.groq_retry.time.sleep"):
            result = call_with_backoff(fn)
        assert result == "ok"
        assert fn.call_count == 2

    def test_raises_runtime_error_after_max_retries(self):
        err = _connection_error()
        fn = MagicMock(side_effect=[err] * _MAX_RETRIES)
        with patch("backend.app.services.groq_retry.time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                call_with_backoff(fn)
        assert "connection" in str(exc_info.value).lower()

    def test_connection_error_also_sleeps_with_backoff(self):
        err = _connection_error()
        fn = MagicMock(side_effect=[err, err, "ok"])
        with patch("backend.app.services.groq_retry.time.sleep") as mock_sleep:
            call_with_backoff(fn)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [_BASE_DELAY * (2 ** 0), _BASE_DELAY * (2 ** 1)]


class TestBadRequestNoRetry:
    def test_bad_request_re_raised_immediately(self):
        """BadRequestError must propagate on attempt 0 — never retried."""
        err = _bad_request()
        fn = MagicMock(side_effect=err)
        with pytest.raises(BadRequestError):
            call_with_backoff(fn)
        assert fn.call_count == 1

    def test_bad_request_not_wrapped_in_runtime_error(self):
        err = _bad_request()
        fn = MagicMock(side_effect=err)
        try:
            call_with_backoff(fn)
        except BadRequestError:
            pass  # expected — correct
        except RuntimeError:
            pytest.fail("BadRequestError must not be wrapped in RuntimeError")

    def test_bad_request_no_sleep(self):
        err = _bad_request()
        fn = MagicMock(side_effect=err)
        with patch("backend.app.services.groq_retry.time.sleep") as mock_sleep:
            with pytest.raises(BadRequestError):
                call_with_backoff(fn)
        mock_sleep.assert_not_called()


class TestAPIStatusErrorNoRetry:
    def test_api_status_error_raises_runtime_immediately(self):
        err = _api_status(500)
        fn = MagicMock(side_effect=err)
        with pytest.raises(RuntimeError, match="API error"):
            call_with_backoff(fn)
        assert fn.call_count == 1

    def test_api_status_error_includes_status_code(self):
        err = _api_status(503)
        fn = MagicMock(side_effect=err)
        with pytest.raises(RuntimeError) as exc_info:
            call_with_backoff(fn)
        assert "503" in str(exc_info.value)

    def test_api_status_error_no_sleep(self):
        err = _api_status(500)
        fn = MagicMock(side_effect=err)
        with patch("backend.app.services.groq_retry.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                call_with_backoff(fn)
        mock_sleep.assert_not_called()


class TestMixedErrors:
    def test_rate_limit_then_connection_then_success(self):
        fn = MagicMock(side_effect=[_rate_limit(), _connection_error(), "final"])
        with patch("backend.app.services.groq_retry.time.sleep"):
            result = call_with_backoff(fn)
        assert result == "final"

    def test_bad_request_after_rate_limit_propagates(self):
        """If first call rate-limits but second call is a BadRequest, propagate BadRequest."""
        fn = MagicMock(side_effect=[_rate_limit(), _bad_request()])
        with patch("backend.app.services.groq_retry.time.sleep"):
            with pytest.raises(BadRequestError):
                call_with_backoff(fn)
