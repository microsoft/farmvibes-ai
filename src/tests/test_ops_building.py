# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import List

import pytest

from vibe_agent.ops import OperationFactory
from vibe_agent.storage.asset_management import LocalFileAssetManager
from vibe_common.constants import DEFAULT_OPS_DIR
from vibe_common.secret_provider import AzureSecretProvider
from vibe_dev.testing.op_tester import FakeStorage


@pytest.fixture
def fake_storage(tmp_path: Path) -> FakeStorage:
    asset_manager = LocalFileAssetManager(str(tmp_path))
    storage = FakeStorage(asset_manager)
    return storage


def test_all_ops_pass_sanity_check(fake_storage: FakeStorage):
    not_sane = [FileNotFoundError, RuntimeError]
    factory = OperationFactory(fake_storage, AzureSecretProvider())
    failures: List[str] = []
    for dirpath, _, filenames in os.walk(DEFAULT_OPS_DIR):
        for fn in filenames:
            if not fn.endswith(".yaml"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                factory.build(path)
            except Exception as e:
                if any([isinstance(e, n) for n in not_sane]):
                    failures.append(fn)
                    print(f"Failed to build op {fn} due to {type(e)}: {e}")
    assert not failures, f"Failed to build the following op(s): {', '.join(failures)}"
