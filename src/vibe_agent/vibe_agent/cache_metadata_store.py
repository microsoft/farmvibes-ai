# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from typing import Dict, Protocol, Set

from hydra_zen import builds
from redis.asyncio import Redis
from redis.asyncio.retry import Retry as RedisRetry
from redis.backoff import DEFAULT_BASE, DEFAULT_CAP, ExponentialBackoff  # type: ignore
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError

from vibe_common.schemas import OpRunId
from vibe_common.secret_provider import retrieve_dapr_secret


class CacheMetadataStoreProtocol(Protocol):
    """
    Protocol for a cache metadata store. This store is used to store and retrieve metadata about
    the relationships of the data (i.e. workflow runs, operation runs and assets) in the cache.
    """

    async def store_references(self, run_id: str, op_run_id: OpRunId, assets: Set[str]) -> None: ...

    async def get_run_ops(self, run_id: str) -> Set[OpRunId]: ...

    async def get_op_workflow_runs(self, op_ref: OpRunId) -> Set[str]: ...

    async def get_op_assets(self, op_ref: OpRunId) -> Set[str]: ...

    async def get_assets_refs(self, asset_ids: Set[str]) -> Dict[str, Set[OpRunId]]: ...

    async def remove_workflow_op_refs(
        self, workflow_run_id: str, op_run_ref: OpRunId
    ) -> Set[str]: ...

    async def remove_op_asset_refs(self, op_run_ref: OpRunId, asset_ids: Set[str]) -> None: ...


class RedisCacheMetadataStore(CacheMetadataStoreProtocol):
    """
    Redis implementation of the cache metadata store.
    """

    # TODO: pass redis service name, namespace, and port through Terraform...
    _redis_host = "redis-master.default.svc.cluster.local"
    _redis_port = 6379
    _key_delimiter = ":"
    _run_ops_key_format = "run:{run_id}:ops"
    _op_runs_key_format = "op:{op_name}:{op_hash}:runs"
    _op_assets_key_format = "op:{op_name}:{op_hash}:assets"
    _asset_ops_key_format = "asset:{asset_id}:ops"
    _op_ref_format = "{op_name}:{op_hash}"

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.redis_password = retrieve_dapr_secret("kubernetes", "redis", "redis-password")

    async def _get_redis_client(self):
        self.logger.debug(
            f"Creating Redis client with host {self._redis_host} and port {self._redis_port}"
        )
        retry = RedisRetry(ExponentialBackoff(cap=DEFAULT_CAP, base=DEFAULT_BASE), 3)
        redis_client = Redis(
            host=self._redis_host,
            port=self._redis_port,
            db=0,
            password=self.redis_password,
            decode_responses=True,
            retry=retry,
            retry_on_error=[ConnectionError, TimeoutError, BusyLoadingError],
        )  # type: ignore
        response = await redis_client.ping()
        self.logger.debug(f"Created redis client - ping response: {response}")
        return redis_client

    def _op_run_id_to_op_ref_str(self, op_run_id: OpRunId) -> str:
        return self._op_ref_format.format(op_name=op_run_id.name, op_hash=op_run_id.hash)

    def _str_to_op_run_id(self, op_run_ref_str: str) -> OpRunId:
        op_name, op_hash = op_run_ref_str.split(self._key_delimiter)
        return OpRunId(name=op_name, hash=op_hash)

    async def store_references(self, run_id: str, op_run_id: OpRunId, assets: Set[str]) -> None:
        # TODO: is a new client needed for every operation or can we intiate in init and reuse?
        redis_client = await self._get_redis_client()

        try:
            pipe = redis_client.pipeline(transaction=True)

            run_ops_key = self._run_ops_key_format.format(run_id=run_id)
            op_ref = self._op_run_id_to_op_ref_str(op_run_id)
            pipe.sadd(run_ops_key, op_ref)

            op_runs_key = self._op_runs_key_format.format(
                op_name=op_run_id.name, op_hash=op_run_id.hash
            )
            pipe.sadd(op_runs_key, run_id)

            if assets:
                op_assets_key = self._op_assets_key_format.format(
                    op_name=op_run_id.name, op_hash=op_run_id.hash
                )
                pipe.sadd(op_assets_key, *assets)

                for asset_id in assets:
                    asset_ops_key = self._asset_ops_key_format.format(asset_id=asset_id)
                    pipe.sadd(asset_ops_key, op_ref)

            await pipe.execute()
            self.logger.debug(
                f"Transaction complete for storing references for run id {run_id} "
                f"(op name {op_run_id.name}, op hash {op_run_id.hash})."
            )
        finally:
            await redis_client.close()

    async def get_run_ops(self, run_id: str) -> Set[OpRunId]:
        """
        Given a workflow run_id, return the set of op run references associated with that workflow
        run as strings in the format "{op_name}:{op_hash}".

        :param run_id: The workflow run id

        :return: The set of op runs associated with the workflow run in the format
            "{op_name}:{op_hash}"
        """
        redis_client = await self._get_redis_client()
        try:
            run_ops_key = self._run_ops_key_format.format(run_id=run_id)
            run_ops = await redis_client.smembers(run_ops_key)
            return {self._str_to_op_run_id(o) for o in run_ops}
        finally:
            await redis_client.close()

    async def get_op_workflow_runs(self, op_run_id: OpRunId) -> Set[str]:
        """
        Given an op run reference, return the set of workflow run ids associated with the op run.

        :param op_ref: The op run reference

        :return: The set of workflow run ids associated with the op run
        """
        redis_client = await self._get_redis_client()
        try:
            op_runs_key = self._op_runs_key_format.format(
                op_name=op_run_id.name, op_hash=op_run_id.hash
            )
            return await redis_client.smembers(op_runs_key)
        finally:
            await redis_client.close()

    async def get_op_assets(self, op_ref: OpRunId) -> Set[str]:
        """
        Given an op run reference, return the set of asset ids associated with the op run.

        :param op_ref: The op run reference

        :return: The set of asset ids associated with the op run
        """
        redis_client = await self._get_redis_client()
        try:
            op_assets_key = self._op_assets_key_format.format(
                op_name=op_ref.name, op_hash=op_ref.hash
            )
            return await redis_client.smembers(op_assets_key)
        finally:
            await redis_client.close()

    async def get_assets_refs(self, asset_ids: Set[str]) -> Dict[str, Set[OpRunId]]:
        """
        Given a list of asset ids, return the set of op run references associated with each asset.

        :param op_ref: The list of asset ids

        :return: A dictionary mapping asset ids to the set of op run references associated with
            each asset
        """
        redis_client = await self._get_redis_client()

        try:
            pipe = redis_client.pipeline(transaction=False)
            asset_ids_list = list(asset_ids)

            for asset_id in asset_ids_list:
                asset_ops_key = self._asset_ops_key_format.format(asset_id=asset_id)
                pipe.smembers(asset_ops_key)

            assets_smembers_result = await pipe.execute()

            results = {}

            for asset_id, asset_smembers in zip(asset_ids_list, assets_smembers_result):
                results[asset_id] = [self._str_to_op_run_id(o) for o in asset_smembers]

            return results
        finally:
            await redis_client.close()

    async def remove_workflow_op_refs(self, workflow_run_id: str, op_run_ref: OpRunId) -> None:
        """
        Removes the references between a workflow run and op run.

        :param workflow_run_id: The workflow run id
        :param op_ref: The op run reference
        """
        redis_client = await self._get_redis_client()
        try:
            pipe = redis_client.pipeline(transaction=True)
            run_ops_key = self._run_ops_key_format.format(run_id=workflow_run_id)
            op_ref = self._op_ref_format.format(op_name=op_run_ref.name, op_hash=op_run_ref.hash)
            pipe.srem(run_ops_key, op_ref)

            op_runs_key = self._op_runs_key_format.format(
                op_name=op_run_ref.name, op_hash=op_run_ref.hash
            )
            pipe.srem(op_runs_key, workflow_run_id)

            await pipe.execute()
            # TODO: check response for number of members removed and emit warning if not 1
        finally:
            await redis_client.close()

    async def remove_op_asset_refs(self, op_run_id: OpRunId, asset_ids: Set[str]) -> None:
        # TODO: the following commands could likely be more efficiently performed by invoking a Lua
        # script that retrieves the op run, iterates through all of the assets ids and removes the
        # asset --> op references and then deletes the op key as well
        redis_client = await self._get_redis_client()
        try:
            pipe = redis_client.pipeline(transaction=True)
            op_assets_key = self._op_assets_key_format.format(
                op_name=op_run_id.name, op_hash=op_run_id.hash
            )

            for asset_id in asset_ids:
                pipe.srem(op_assets_key, asset_id)

                asset_ops_key = self._asset_ops_key_format.format(asset_id=asset_id)
                op_run_ref = self._op_ref_format.format(
                    op_name=op_run_id.name, op_hash=op_run_id.hash
                )
                pipe.srem(asset_ops_key, op_run_ref)

            await pipe.execute()
            # TODO: check response for number removed and emit warning if doesn't make sense
        finally:
            await redis_client.close()


CacheMetadataStoreProtocolConfig = builds(
    CacheMetadataStoreProtocol,
)

RedisCacheMetadataStoreConfig = builds(
    RedisCacheMetadataStore,
    builds_bases=(CacheMetadataStoreProtocolConfig,),
    # config={"redis_url": getenv("REDIS_URL", "redis://localhost:6379")}
)
