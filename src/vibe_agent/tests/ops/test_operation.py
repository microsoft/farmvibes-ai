import os
from datetime import datetime
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
from shapely import geometry as shpg

from vibe_agent.ops import Operation, OperationFactory
from vibe_agent.ops_helper import OpIOConverter
from vibe_agent.storage.local_storage import LocalResourceExistsError
from vibe_common.schemas import CacheInfo, OperationParser
from vibe_core.data import DataVibe
from vibe_core.data.utils import StacConverter
from vibe_dev.testing.fake_workflows_fixtures import fake_ops_dir  # type: ignore # noqa
from vibe_dev.testing.op_tester import FakeStorage


@patch.object(OperationFactory, "resolve_secrets")
def test_callback_output_mismatch_fails(resolve_secrets: MagicMock, fake_ops_dir: str):  # noqa
    op_spec = OperationParser().parse(os.path.join(fake_ops_dir, "fake/item_item.yaml"))
    resolve_secrets.return_value = op_spec.parameters
    factory = OperationFactory(None, None)  # type: ignore
    op = factory.build(op_spec)

    now = datetime.now()
    x = DataVibe(
        id="1", time_range=(now, now), geometry=shpg.mapping(shpg.box(0, 0, 1, 1)), assets=[]
    )
    op._call_validate_op(user_data=x)  # type: ignore

    def mock_callback(callback: Callable[..., Any]):
        def fun(*args: Any, **kwargs: Any):
            return {"wrong": None, **callback(*args, **kwargs)}

        return fun

    op.callback = mock_callback(op.callback)  # type: ignore
    with pytest.raises(RuntimeError):
        op._call_validate_op(user_data=x)  # type: ignore


@patch.object(Operation, "_call_validate_op")
@patch.object(FakeStorage, "retrieve_output_from_input_if_exists")
@patch.object(OpIOConverter, "serialize_output")
@patch.object(OpIOConverter, "deserialize_input")
@patch.object(OperationFactory, "resolve_secrets")
def test_op_cache_check_before_callback(
    resolve_secrets: MagicMock,
    deserialize_input: MagicMock,
    serialize_output: MagicMock,
    retrieve_cache: MagicMock,
    call_validate: MagicMock,
    fake_ops_dir: str,  # noqa
):
    deserialize_input.return_value = {"stac": 1}
    serialize_output.side_effect = lambda x: x
    cached_output = {"cached_before": "no callback 😊"}
    retrieve_cache.return_value = cached_output
    op_spec = OperationParser().parse(os.path.join(fake_ops_dir, "fake/item_item.yaml"))
    resolve_secrets.return_value = op_spec.parameters
    factory = OperationFactory(FakeStorage(None), None)  # type:ignore
    op = factory.build(op_spec)
    cache_info = CacheInfo("test-op", "1.0", {}, {})
    object.__setattr__(cache_info, "hash", "cache_before")
    out = op.run(None, cache_info)  # type:ignore
    assert out == cached_output
    deserialize_input.assert_called_once()
    serialize_output.assert_called_once()
    retrieve_cache.assert_called_once()
    call_validate.assert_not_called()


@patch.object(FakeStorage, "store")
@patch.object(StacConverter, "from_stac_item")
@patch.object(Operation, "_call_validate_op")
@patch.object(FakeStorage, "retrieve_output_from_input_if_exists")
@patch.object(OpIOConverter, "serialize_output")
@patch.object(OpIOConverter, "deserialize_input")
@patch.object(OperationFactory, "resolve_secrets")
def test_op_cache_check_after_callback(
    resolve_secrets: MagicMock,
    deserialize_input: MagicMock,
    serialize_output: MagicMock,
    retrieve_cache: MagicMock,
    call_validate: MagicMock,
    from_stac_item: MagicMock,
    store: MagicMock,
    fake_ops_dir: str,  # noqa
):
    deserialize_input.return_value = {"stac": 1}
    serialize_output.side_effect = lambda x: x
    cached_output = {"cached_after": "yes callback 😔"}
    retrieve_cache.side_effect = [None, cached_output]
    call_validate.return_value = {"out": "repeated callback output"}
    from_stac_item.side_effect = lambda x: x
    store.side_effect = LocalResourceExistsError()
    op_spec = OperationParser().parse(os.path.join(fake_ops_dir, "fake/item_item.yaml"))
    resolve_secrets.return_value = op_spec.parameters
    factory = OperationFactory(FakeStorage(None), None)  # type:ignore
    op = factory.build(op_spec)
    cache_info = CacheInfo("test-op", "1.0", {}, {})
    object.__setattr__(cache_info, "hash", "cache_before")
    out = op.run(None, cache_info)  # type:ignore
    assert out == cached_output
    deserialize_input.assert_called_once()
    serialize_output.assert_called_once()
    # Cache retrieval should be called once before the callback, and then again after
    assert retrieve_cache.call_count == 2
    call_validate.assert_called_once()
