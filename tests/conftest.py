import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast_ai.data import synthetic_series  # noqa: E402


@pytest.fixture(scope="session")
def sales():
    return synthetic_series("sales", days=600, seed=7)
