# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Set, Tuple
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from vibe_agent.cache_metadata_store import RedisCacheMetadataStore
from vibe_agent.data_ops import DataOpsManager
from vibe_agent.storage import asset_management
from vibe_agent.storage.storage import Storage
from vibe_common.schemas import CacheInfo, OpRunId
from vibe_core.data.core_types import OpIOType
from vibe_core.datamodel import RunConfig, RunDetails, RunStatus, SpatioTemporalJson


class FakeOpRunResult:
    def __init__(self, op_name: str, fake_asset_ids: Set[str]):
        self.cache_info = CacheInfo(op_name, "1.0", {}, {})
        self.asset_ids = fake_asset_ids

    def get_output(self) -> OpIOType:
        return {self.cache_info.name: {"assets": {asset_id: {} for asset_id in self.asset_ids}}}

    def get_op_run_id(self) -> OpRunId:
        return OpRunId(self.cache_info.name, self.cache_info.hash)


@pytest.fixture
def no_asset_op_run():
    return FakeOpRunResult("no_asset_op", set())


@pytest.fixture
def op_1_run():
    return FakeOpRunResult("op_1_run", {"asset-1", "asset-2"})


@pytest.fixture
def op_2_run():
    return FakeOpRunResult("op_2_run", {"asset-2", "asset-3"})


@pytest.fixture
def run_config() -> Dict[str, Any]:
    run_config = asdict(
        RunConfig(
            name="fake",
            workflow="fake",
            parameters=None,
            user_input=SpatioTemporalJson(
                datetime.now(),
                datetime.now(),
                {},
            ),
            id=uuid.uuid4(),
            details=RunDetails(),
            task_details={},
            spatio_temporal_json=None,
            output="",
        )
    )
    return run_config


class AsyncFakeRedis:
    def __init__(self):
        self.data = {}

    async def sadd(self, key: str, *values: str):
        if key not in self.data:
            self.data[key] = set()
        self.data[key].update(values)

    async def srem(self, key: str, *values: str):
        if key in self.data:
            self.data[key].difference_update(values)
        # Redis does not allow empty sets
        if not self.data[key]:
            del self.data[key]

    async def smembers(self, key: str):
        return self.data.get(key, set())

    async def scard(self, key: str):
        return len(self.data.get(key, set()))

    async def sismember(self, key: str, value: str):
        return value in self.data.get(key, set())

    def pipeline(self, transaction: bool = True):
        return AsyncFakeRedisPipeline(self)

    async def close(self):
        pass


class AsyncFakeRedisPipeline:
    def __init__(self, redis_client: AsyncFakeRedis):
        self.redis_client = redis_client
        self.commands = []

    def __getattr__(self, name: str):
        def method(*args: Any, **kwargs: Any):
            command = (name, args, kwargs)
            self.commands.append(command)

        return method

    async def execute(self):
        coroutines = []
        for command in self.commands:
            name, args, kwargs = command
            method = getattr(self.redis_client, name)
            coro = method(*args, **kwargs)
            coroutines.append(coro)
        results = await asyncio.gather(*coroutines)
        return results


def get_mocked_data_ops() -> Tuple[DataOpsManager, AsyncFakeRedis, Mock]:
    with patch("vibe_agent.cache_metadata_store.retrieve_dapr_secret"):
        redis_client_mock = AsyncFakeRedis()

        storage_mock = Mock(spec=Storage)
        storage_mock.asset_manager = Mock(spec=asset_management.AssetManager)

        metadata_store = RedisCacheMetadataStore()
        metadata_store._get_redis_client = AsyncMock(return_value=redis_client_mock)

        do_manager = DataOpsManager(storage_mock, metadata_store=metadata_store)
        do_manager._init_locks()
        return do_manager, redis_client_mock, storage_mock


def assert_op_in_fake_redis(redis_client: AsyncFakeRedis, run_id: str, fake_op: FakeOpRunResult):
    run_ops_key = RedisCacheMetadataStore._run_ops_key_format.format(run_id=run_id)
    op_runs_key = RedisCacheMetadataStore._op_runs_key_format.format(
        op_name=fake_op.cache_info.name, op_hash=fake_op.cache_info.hash
    )
    op_ref = RedisCacheMetadataStore._op_ref_format.format(
        op_name=fake_op.cache_info.name, op_hash=fake_op.cache_info.hash
    )
    op_assets_key = RedisCacheMetadataStore._op_assets_key_format.format(
        op_name=fake_op.cache_info.name, op_hash=fake_op.cache_info.hash
    )
    assert redis_client.data[run_ops_key] == {op_ref}
    assert run_id in redis_client.data[op_runs_key]

    if fake_op.asset_ids:
        assert redis_client.data[op_assets_key] == fake_op.asset_ids

        for asset_id in fake_op.asset_ids:
            asset_op_key = RedisCacheMetadataStore._asset_ops_key_format.format(asset_id=asset_id)
            assert op_ref in redis_client.data[asset_op_key]


@pytest.mark.anyio
async def test_store_references_with_empty_asset_list(no_asset_op_run: FakeOpRunResult):
    do_manager, redis_client_mock, _ = get_mocked_data_ops()
    await do_manager.add_references(
        "fake-run", no_asset_op_run.get_op_run_id(), no_asset_op_run.get_output()
    )

    assert_op_in_fake_redis(redis_client_mock, "fake-run", no_asset_op_run)


@pytest.mark.anyio
async def test_store_references_simple(op_1_run: FakeOpRunResult):
    do_manager, redis_client_mock, _ = get_mocked_data_ops()
    await do_manager.add_references("fake-run", op_1_run.get_op_run_id(), op_1_run.get_output())
    assert len(redis_client_mock.data) == 3 + len(op_1_run.asset_ids)
    assert_op_in_fake_redis(redis_client_mock, "fake-run", op_1_run)


@pytest.mark.anyio
async def test_store_references_two_wfs_shared_op(op_1_run: FakeOpRunResult):
    do_manager, redis_client_mock, _ = get_mocked_data_ops()
    await do_manager.add_references("fake-run-1", op_1_run.get_op_run_id(), op_1_run.get_output())
    await do_manager.add_references("fake-run-2", op_1_run.get_op_run_id(), op_1_run.get_output())

    assert len(redis_client_mock.data) == 4 + len(op_1_run.asset_ids)

    assert_op_in_fake_redis(redis_client_mock, "fake-run-1", op_1_run)
    assert_op_in_fake_redis(redis_client_mock, "fake-run-2", op_1_run)


@pytest.mark.anyio
async def test_store_references_two_wfs_shared_asset(
    op_1_run: FakeOpRunResult,
    op_2_run: FakeOpRunResult,
):
    do_manager, redis_client_mock, _ = get_mocked_data_ops()
    await do_manager.add_references("fake-run-1", op_1_run.get_op_run_id(), op_1_run.get_output())
    await do_manager.add_references("fake-run-2", op_2_run.get_op_run_id(), op_2_run.get_output())

    assert len(redis_client_mock.data) == 6 + len(op_1_run.asset_ids) + len(op_2_run.asset_ids) - 1

    assert_op_in_fake_redis(redis_client_mock, "fake-run-1", op_1_run)
    assert_op_in_fake_redis(redis_client_mock, "fake-run-2", op_2_run)


@patch("vibe_common.statestore.StateStore.retrieve")
@pytest.mark.anyio
async def test_delete_invalid_workflow_run(ss_retrieve_mock: Mock, run_config: Dict[str, Any]):
    do_manager, _, _ = get_mocked_data_ops()
    invalid_delete_statuses = [
        RunStatus.pending,
        RunStatus.queued,
        RunStatus.running,
        RunStatus.deleting,
        RunStatus.deleted,
    ]

    for status in invalid_delete_statuses:
        run_config["details"]["status"] = status
        ss_retrieve_mock.return_value = run_config
        result = await do_manager.delete_workflow_run("fake-run")
        assert not result


@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_delete_workflow_run_no_assets(
    ss_store_mock: Mock,
    ss_retrieve_mock: Mock,
    no_asset_op_run: FakeOpRunResult,
    run_config: Dict[str, Any],
):
    do_manager, redis_client_mock, storage_mock = get_mocked_data_ops()
    await do_manager.add_references(
        "fake-run", no_asset_op_run.get_op_run_id(), no_asset_op_run.get_output()
    )

    run_config["details"]["status"] = RunStatus.done
    ss_retrieve_mock.return_value = run_config
    await do_manager.delete_workflow_run("fake-run")

    assert ss_store_mock.call_count == 2
    rc1 = ss_store_mock.call_args_list[0][0][1]
    assert rc1.details.status == RunStatus.deleting
    rc2 = ss_store_mock.call_args_list[1][0][1]
    assert rc2.details.status == RunStatus.deleted

    storage_mock.asset_manager.remove.assert_not_called()
    storage_mock.remove.assert_called_once_with(no_asset_op_run.get_op_run_id())

    assert len(redis_client_mock.data) == 0


@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_delete_workflow_run_simple(
    ss_store_mock: Mock,
    ss_retrieve_mock: Mock,
    op_1_run: FakeOpRunResult,
    run_config: Dict[str, Any],
):
    do_manager, redis_client_mock, storage_mock = get_mocked_data_ops()
    await do_manager.add_references("fake-run", op_1_run.get_op_run_id(), op_1_run.get_output())

    run_config["details"]["status"] = RunStatus.done
    ss_retrieve_mock.return_value = run_config
    await do_manager.delete_workflow_run("fake-run")

    assert ss_store_mock.call_count == 2
    rc1 = ss_store_mock.call_args_list[0][0][1]
    assert rc1.details.status == RunStatus.deleting
    rc2 = ss_store_mock.call_args_list[1][0][1]
    assert rc2.details.status == RunStatus.deleted

    calls = [call(asset_id) for asset_id in op_1_run.asset_ids]
    storage_mock.asset_manager.remove.assert_has_calls(calls, any_order=True)
    storage_mock.remove.assert_called_once_with(op_1_run.get_op_run_id())

    assert len(redis_client_mock.data) == 0


@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_delete_workflow_run_overlapping_op_and_asset(
    ss_store_mock: Mock,
    ss_retrieve_mock: Mock,
    op_1_run: FakeOpRunResult,
    op_2_run: FakeOpRunResult,
    run_config: Dict[str, Any],
):
    do_manager, redis_client_mock, storage_mock = get_mocked_data_ops()
    await do_manager.add_references("fake-run-1", op_1_run.get_op_run_id(), op_1_run.get_output())
    await do_manager.add_references("fake-run-1", op_2_run.get_op_run_id(), op_2_run.get_output())
    await do_manager.add_references("fake-run-2", op_1_run.get_op_run_id(), op_1_run.get_output())

    run_config["details"]["status"] = RunStatus.done
    ss_retrieve_mock.return_value = run_config
    await do_manager.delete_workflow_run("fake-run-1")

    storage_mock.asset_manager.remove.assert_called_once_with("asset-3")
    storage_mock.remove.assert_called_once_with(op_2_run.get_op_run_id())

    assert_op_in_fake_redis(redis_client_mock, "fake-run-2", op_1_run)
    assert len(redis_client_mock.data) == 3 + len(op_1_run.asset_ids)
