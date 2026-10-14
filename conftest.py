import hashlib
import os
from pathlib import Path
import tempfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
project_id = hashlib.sha256(str(PROJECT_ROOT).encode()).hexdigest()[:12]
os.environ["CODEAGENT_TEST_CACHE"] = str(Path(tempfile.gettempdir()) / f"codeagent-pytest-cache-{project_id}")


def pytest_configure(config):
    # Use a unique system directory; cleanup runs after all fixtures finish.
    if config.option.basetemp:
        if Path(config.option.basetemp).resolve().is_relative_to(PROJECT_ROOT):
            raise pytest.UsageError("--basetemp must stay outside the repository.")
    else:
        temporary = tempfile.TemporaryDirectory(prefix="codeagent-pytest-")
        config.option.basetemp = temporary.name
        config.add_cleanup(temporary.cleanup)


@pytest.fixture
def anyio_backend():
    return "asyncio"
