# flake8: noqa
import os
import uuid
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest

from vibe_agent.ops import OperationFactoryConfig
from vibe_agent.worker import Worker
from vibe_dev.testing.storage_fixtures import *  # type: ignore # noqa: F403, F401
from vibe_dev.testing import anyio_backend  # type: ignore # noqa
from vibe_dev.testing.workflow_fixtures import (
    SimpleStrData,
    simple_op_spec,
    workflow_execution_message,
)  # type: ignore # noqa

FILE_CONTENTS = "SAMPLE FILE CONTENTS FOR TESTING PURPOSES"


@pytest.fixture(scope="module")
def file_contents():
    return FILE_CONTENTS


@pytest.fixture(scope="module")
def local_file(file_contents: str):
    with TemporaryDirectory() as tmp_dir:
        filename = f"{uuid.uuid4()}.txt"
        filepath = os.path.join(tmp_dir, filename)
        with open(os.path.join(tmp_dir, filename), "w") as f:
            f.write(file_contents)
        yield filepath


@pytest.fixture
def local_file_ref(request: pytest.FixtureRequest, local_file: str):
    ref_type: str = request.param  # type: ignore
    if ref_type == "uri":
        return f"file://{local_file}"
    elif ref_type == "path":
        return local_file
    else:
        raise ValueError(f"Invalid reference type {ref_type}")


@pytest.fixture
def op_yaml() -> Dict[str, Any]:
    return {
        "name": "fake",
        "inputs": {
            "user_data": "List[DataVibe]",
        },
        "output": {
            "processed_data": "List[DataVibe]",
        },
        "parameters": {
            "fake_param": 1,
            "fake_another_param": {"fake_nested": 2, "fake_nested_too": 3},
        },
        "entrypoint": {"file": "op.py", "callback_builder": "callback_builder"},
    }


@pytest.fixture
def op_foo() -> str:
    foo_str: str = """
def print_args(user_data):
  return user_data

def callback_builder(**kw):
  return print_args
    """
    return foo_str


@pytest.fixture
def non_existing_file(request: pytest.FixtureRequest):
    location = request.param  # type:ignore
    if location == "local":
        return "/nodir/nodir2/does_not_exist.txt"
    raise ValueError(f"Expected 'local' or 'remote' request, got {location}")
