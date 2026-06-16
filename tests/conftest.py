import os
from unittest.mock import MagicMock, patch

# Must be set before any backend import that touches Groq
os.environ.setdefault("GROQ_API_KEY", "test-key-dummy")

# Patch Groq at the package level so no real network calls are made
_groq_patcher = patch("groq.Groq", return_value=MagicMock())
_groq_patcher.start()
