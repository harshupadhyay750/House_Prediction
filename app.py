"""Streamlit entry point for deployment on Streamlit Community Cloud."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "app" / "app.py"

if not APP_FILE.exists():
    raise FileNotFoundError(f"Expected Streamlit app file not found: {APP_FILE}")

spec = importlib.util.spec_from_file_location("propintel_streamlit_app", APP_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Streamlit app module from {APP_FILE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
