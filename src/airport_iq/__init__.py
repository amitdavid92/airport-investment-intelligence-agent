"""Airport Investment Intelligence Agent.

Loads `.env` from the project root on import so the API server, the Streamlit
UI and pytest all see the same credentials without needing them exported in
whichever shell happens to launch them.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# `override=False`: a variable already exported in the environment wins over
# the file, so a one-off `ANTHROPIC_API_KEY=... uv run ...` still works.
load_dotenv(PROJECT_ROOT / ".env", override=False)
