import os
from dataclasses import dataclass

import pytest

from vibe_core.data.core_types import BaseVibe

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(HERE, "fake_workflows")
OPS_DIR = os.path.join(HERE, "fake_ops")


@dataclass
class FakeType(BaseVibe):
    data: str


def get_fake_workflow_path(workflow_name: str):
    return os.path.join(WORKFLOWS_DIR, f"{workflow_name}.yaml")


@pytest.fixture
def fake_workflow_path(request: pytest.FixtureRequest):
    workflow_name = request.param  # type:ignore
    return get_fake_workflow_path(workflow_name)


@pytest.fixture
def fake_ops_dir() -> str:
    return OPS_DIR


@pytest.fixture
def fake_workflows_dir() -> str:
    return WORKFLOWS_DIR
