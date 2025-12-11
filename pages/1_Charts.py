"""Charts page - wrapper for Streamlit Cloud."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.pages.charts import *  # noqa: F401, F403
