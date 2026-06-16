import logging
import time
from typing import Callable, TypeVar

from groq import RateLimitError, APIConnectionError, APIStatusError, BadRequestError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_RETRIES = 3
_BASE_DELAY = 1.5  # seconds


def call_with_backoff(fn: Callable[[], T], label: str = "") -> T:
    """
    Call fn() with exponential backoff on RateLimitError / APIConnectionError.
    Raises RuntimeError on terminal failures.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except RateLimitError as e:
            delay = _BASE_DELAY * (2 ** attempt)
            if attempt < _MAX_RETRIES - 1:
                logger.warning("%s rate-limited, retrying in %.1fs (attempt %d/%d)",
                               label, delay, attempt + 1, _MAX_RETRIES)
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Groq rate limit exceeded after {_MAX_RETRIES} retries — "
                    f"please wait and retry: {e}"
                ) from e
        except APIConnectionError as e:
            delay = _BASE_DELAY * (2 ** attempt)
            if attempt < _MAX_RETRIES - 1:
                logger.warning("%s connection error, retrying in %.1fs (attempt %d/%d)",
                               label, delay, attempt + 1, _MAX_RETRIES)
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Groq connection error after {_MAX_RETRIES} retries — "
                    f"check network/API key: {e}"
                ) from e
        except BadRequestError as e:
            raise RuntimeError(f"Groq bad request: {e}") from e
        except APIStatusError as e:
            raise RuntimeError(f"Groq API error {e.status_code}: {e.message}") from e
    raise RuntimeError(f"call_with_backoff: exhausted {_MAX_RETRIES} attempts")
