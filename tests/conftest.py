import os
from unittest.mock import MagicMock, patch

# Dummy keys so unit tests can import agent modules without RuntimeError.
# Integration tests (test_video_pipeline.py) require real keys and skip if absent.
os.environ.setdefault("GROQ_API_KEY", "test-key-dummy")
os.environ.setdefault("GREENNODE_AIP_KEY", "test-key-dummy-unit")
os.environ.setdefault("GREENNODE_AIP_BASE_URL", "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1")

# Keep Groq patched for tests that still import groq exceptions (test_groq_retry.py)
try:
    _groq_patcher = patch("groq.Groq", return_value=MagicMock())
    _groq_patcher.start()
except Exception:
    pass
