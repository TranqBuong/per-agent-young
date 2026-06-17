import logging
import re
import time
from typing import Callable, TypeVar

from openai import RateLimitError, APIConnectionError, APIStatusError, BadRequestError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_RETRIES = 3
_BASE_DELAY = 5.0    # base wait when no retry-after header (5s → 10s → give up)
_MAX_AUTO_WAIT = 360  # don't auto-sleep longer than 6 min


def _parse_retry_after(message: str) -> float:
    """Parse 'Please try again in 3m49.478s' → seconds. Returns 0.0 if not found."""
    m = re.search(r'[Pp]lease try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', str(message))
    if m:
        minutes = int(m.group(1) or 0)
        secs = float(m.group(2))
        return minutes * 60 + secs + 2.0
    return 0.0


def call_with_backoff(fn: Callable[[], T], label: str = "") -> T:
    """
    Call fn() with exponential backoff on RateLimitError / APIConnectionError.
    Raises RuntimeError on terminal failures.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except RateLimitError as e:
            wait = _parse_retry_after(str(e))
            if wait > 0 and wait <= _MAX_AUTO_WAIT and attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "%s rate-limited (429), waiting %.0fs as suggested (attempt %d/%d)",
                    label, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
            elif attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("%s rate-limited, retrying in %.1fs (attempt %d/%d)",
                               label, delay, attempt + 1, _MAX_RETRIES)
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Rate limit exceeded after {_MAX_RETRIES} retries — "
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
                    f"Connection error after {_MAX_RETRIES} retries — "
                    f"check network/API key: {e}"
                ) from e
        except BadRequestError:
            raise
        except APIStatusError as e:
            if e.status_code == 429 and attempt < _MAX_RETRIES - 1:
                wait = _parse_retry_after(str(e))
                delay = wait if 0 < wait <= _MAX_AUTO_WAIT else _BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "%s API status 429, waiting %.0fs (attempt %d/%d)",
                    label, delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"API error {e.status_code}: {e.message}") from e
    raise RuntimeError(f"call_with_backoff: exhausted {_MAX_RETRIES} attempts")
