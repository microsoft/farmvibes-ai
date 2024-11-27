# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
from typing import List, cast

import pytest

from vibe_core.data import BaseVibeDict, DataVibe
from vibe_core.testing.comparison import assert_all_close
from vibe_dev.testing.op_tester import OpTester, ReferenceRetriever

HERE = os.path.dirname(os.path.abspath(__file__))
OPS_DIR = os.path.join(HERE, "..", "..", "..", "ops")
FILES_DIR = "/tmp/op_references/"
TEST_OPS = [
    "compute_cloud_prob",
    "compute_sentinel_shadow",
    "download_sentinel_1",
    "download_sentinel_2_from_gcp",
    "download_sentinel_2_from_pc",
    "filter_items",
    "list_sentinel_1_products",
    "list_sentinel_2_L1C",
    "list_sentinel_2_L2A",
    "merge_cloud_masks",
    "merge_sentinel1_orbits",
    "merge_sentinel_orbits",
    "preprocess_sentinel1",
    "preprocess_sentinel2",
]
OP_YAML_DIR = {
    "list_sentinel_2_L1C": "list_sentinel_2_products",
    "list_sentinel_2_L2A": "list_sentinel_2_products",
}


@pytest.fixture
def reference_retriever():
    return ReferenceRetriever(FILES_DIR)


@pytest.fixture
def op_tester(request: pytest.FixtureRequest):
    op_name: str = request.param  # type: ignore
    op_dir = OP_YAML_DIR.get(op_name, op_name)
    op_config_path = os.path.join(OPS_DIR, op_dir, f"{op_name}.yaml")
    return OpTester(op_config_path)


@pytest.fixture
def test_data(request: pytest.FixtureRequest, reference_retriever: ReferenceRetriever):
    op_name = request.param  # type: ignore
    return reference_retriever.retrieve(op_name)


@pytest.mark.parametrize("op_tester,test_data", [(t, t) for t in TEST_OPS], indirect=True)
def test_op_outputs(op_tester: OpTester, test_data: List[List[BaseVibeDict]]):
    for input_data, expected_output in test_data:
        start = time.time()
        op_output = op_tester.run(**input_data)
        end = time.time()
        for name, out in op_output.items():
            expected = expected_output[name]
            if isinstance(expected, list):
                sort_expected = sorted(expected, key=lambda x: x.time_range[0])
                sort_out = sorted(cast(List[DataVibe], out), key=lambda x: x.time_range[0])
                for o1, o2 in zip(sort_expected, sort_out):
                    assert_all_close(o1, o2)
            else:
                assert isinstance(out, DataVibe)
                assert_all_close(expected, out)
        print(f"Spent {end - start}s on op: {op_tester.op.name}")
