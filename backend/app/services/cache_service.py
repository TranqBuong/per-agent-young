import hashlib
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STORAGE = Path(__file__).parent.parent.parent.parent / "storage"
_TTL_SECONDS = 2 * 24 * 3600  # 2 days
_SIMILARITY_THRESHOLD = 0.85


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode()).hexdigest()[:16]


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class CacheService:
    def __init__(self):
        _STORAGE.mkdir(parents=True, exist_ok=True)

    # ── Cleanup expired entries ───────────────────────────────────────
    def cleanup(self) -> int:
        now = time.time()
        removed = 0
        for f in _STORAGE.glob("cache_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if now - data.get("created_at", 0) > _TTL_SECONDS:
                    f.unlink()
                    removed += 1
            except Exception as exc:
                logger.warning("Cache cleanup: could not process %s — %s", f.name, exc)
        if removed:
            logger.info("Cache cleanup: removed %d expired file(s)", removed)
        return removed

    # ── Lookup: exact hash first, similarity fallback ─────────────────
    def get(self, key: str, requirement_text: str) -> Optional[Any]:
        self.cleanup()
        now = time.time()
        # Fast path: exact hash match
        h = _hash(requirement_text)
        exact = _STORAGE / f"cache_{key}_{h}.json"
        if exact.exists():
            try:
                data = json.loads(exact.read_text(encoding="utf-8"))
                if now - data.get("created_at", 0) <= _TTL_SECONDS:
                    logger.info("Cache hit (exact) key=%s", key)
                    return data.get("response")
            except Exception as exc:
                logger.warning("Cache read error %s — %s", exact.name, exc)

        # Slow path: similarity scan
        best_score, best_response = 0.0, None
        for f in _STORAGE.glob(f"cache_{key}_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if now - data.get("created_at", 0) > _TTL_SECONDS:
                    continue
                score = _similarity(requirement_text, data.get("requirement", ""))
                if score >= _SIMILARITY_THRESHOLD and score > best_score:
                    best_score = score
                    best_response = data.get("response")
            except Exception as exc:
                logger.warning("Cache read error %s — %s", f.name, exc)

        if best_response is not None:
            logger.info("Cache hit (similarity=%.2f) key=%s", best_score, key)
        return best_response

    # ── Save response ─────────────────────────────────────────────────
    def set(self, key: str, requirement_text: str, response: Any) -> None:
        self.cleanup()
        h = _hash(requirement_text)
        path = _STORAGE / f"cache_{key}_{h}.json"
        try:
            path.write_text(
                json.dumps(
                    {
                        "key": key,
                        "requirement": requirement_text,
                        "created_at": time.time(),
                        "response": response,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Cache saved key=%s hash=%s", key, h)
        except Exception as exc:
            logger.error("Cache write failed key=%s — %s", key, exc)
