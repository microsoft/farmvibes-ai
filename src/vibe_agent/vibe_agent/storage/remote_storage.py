# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from hashlib import sha256
from math import ceil
from typing import Any, Dict, List, Optional, cast

from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey
from azure.cosmos.aio import (
    ContainerProxy as AsyncContainerProxy,
)
from azure.cosmos.aio import (
    CosmosClient as AsyncCosmosClient,
)
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.storage.blob import BlobLeaseClient
from hydra_zen import MISSING
from pystac.item import Item

from vibe_common.constants import (
    DEFAULT_COSMOS_DATABASE_NAME,
    DEFAULT_COSMOS_URI,
    DEFAULT_STAC_COSMOS_CONTAINER,
)
from vibe_common.schemas import CacheInfo, OpRunId
from vibe_core.utils import ensure_list

from .asset_management import AssetManager, BlobAssetManagerConfig
from .storage import ItemDict, Storage, StorageConfig

LeaseDict = Dict[str, BlobLeaseClient]


@dataclass
class CosmosData:
    id: str
    op_name: str


@dataclass
class ItemList(CosmosData):
    output_name: str
    items: List[Dict[str, Any]]
    type: str = "item_list"


@dataclass
class RunInfo(CosmosData):
    run_id: str
    cache_info: Dict[str, Any]
    items: List[str]
    singular_items: List[str]
    type: str = "run_info"


class CosmosStorage(Storage):
    PARTITION_KEY = "/op_name"
    LIST_MIN_SIZE: int = 1
    # https://docs.microsoft.com/en-us/rest/api/cosmos-db/http-status-codes-for-cosmosdb
    entity_too_large_status_code: int = 413

    def __init__(
        self,
        key: str,
        asset_manager: AssetManager,
        stac_container_name: str = DEFAULT_STAC_COSMOS_CONTAINER,
        cosmos_database_name: str = DEFAULT_COSMOS_DATABASE_NAME,
        cosmos_url: str = DEFAULT_COSMOS_URI,
        list_max_size: int = 1024,
    ):
        super().__init__(asset_manager)
        self.key = key
        self.cosmos_url = cosmos_url
        self.cosmos_database_name = cosmos_database_name
        self.stac_container_name = stac_container_name
        self.container_proxy_async = None
        self.list_max_size = list_max_size
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @lru_cache
    def container_proxy(self):
        cosmos_client = CosmosClient(self.cosmos_url, self.key)
        db = cosmos_client.create_database_if_not_exists(id=self.cosmos_database_name)
        return db.create_container_if_not_exists(
            self.stac_container_name, partition_key=PartitionKey(self.PARTITION_KEY)
        )

    def _convert_items(self, items: ItemDict):
        converted_items: Dict[str, List[Dict[str, Any]]] = {}
        singular_items: List[str] = []
        for key, item in items.items():
            if isinstance(item, Item):
                singular_items.append(key)
            item = ensure_list(item)
            converted_item = [i.to_dict() for i in item]
            converted_items[key] = converted_item
        return converted_items, singular_items

    def _build_item_list_id(self, ids: List[str], output_name: str, run_hash: str):
        ids.append(run_hash)
        ids.append(output_name)
        return sha256("".join(ids).encode()).hexdigest()

    def _build_items_to_store(
        self,
        op_name: str,
        run_hash: str,
        item_dict: Dict[str, List[Dict[str, Any]]],
        list_size: int,
    ):
        output: List[ItemList] = []
        id_list: List[str] = []
        for output_name, items in item_dict.items():
            items = ensure_list(items)
            num_items = len(items)
            num_partitions = ceil(num_items / list_size)
            for i in range(num_partitions):
                offset = i * list_size
                last_item = min(offset + list_size, num_items)
                partitioned_items = items[offset:last_item]
                items_ids = [i["id"] for i in partitioned_items]
                partition_id = self._build_item_list_id(items_ids, output_name, run_hash)
                id_list.append(partition_id)
                output.append(ItemList(partition_id, op_name, output_name, partitioned_items))
        return output, id_list

    def _store_data(
        self, op_name: str, run_to_store: Dict[str, Any], items_to_store: List[Dict[str, Any]]
    ):
        container = self._get_container()
        stored_items: List[str] = []
        try:
            for i in items_to_store:
                container.create_item(body=i)
                stored_items.append(i["id"])
            container.create_item(body=run_to_store)
        except Exception:
            # rolling back
            for i in stored_items:
                container.delete_item(i, op_name)
            raise

    def store(self, run_id: str, items: ItemDict, cache_info: CacheInfo) -> ItemDict:
        items = self.asset_handler.copy_assets(items)
        dict_items, singular_items = self._convert_items(items)
        extra_fields = cache_info.as_storage_dict()
        run_hash = extra_fields[self.HASH_FIELD]
        list_size = self.list_max_size
        e = RuntimeError("No tries to store have been made")
        items_lists: List[ItemList] = []
        while list_size > self.LIST_MIN_SIZE:
            try:
                items_lists, items_id_list = self._build_items_to_store(
                    cache_info.name, run_hash, dict_items, list_size
                )
                run_to_store = asdict(
                    RunInfo(
                        run_hash,
                        cache_info.name,
                        run_id,
                        extra_fields,
                        items_id_list,
                        singular_items,
                    )
                )
                items_to_store = [asdict(items_list) for items_list in items_lists]
                self._store_data(cache_info.name, run_to_store, items_to_store)
                return items
            except CosmosHttpResponseError as er:
                try:
                    status_code = int(er.status_code)  # type: ignore
                except TypeError:
                    raise er  # Couldn't get the status code, so just break
                if status_code != self.entity_too_large_status_code:
                    # We are only handling EntityTooLarge
                    raise
                e = er
                list_size = ceil(max(len(i.items) for i in items_lists) / 2)
        raise RuntimeError(
            f"Could not store items. Tried from {self.list_max_size} "
            f"to {self.LIST_MIN_SIZE} sized lists"
        ) from e

    def _get_container(self) -> ContainerProxy:
        return self.container_proxy

    def _get_container_async(self) -> AsyncContainerProxy:
        if self.container_proxy_async is None:
            cosmos_client_async = AsyncCosmosClient(self.cosmos_url, self.key)
            db = cosmos_client_async.get_database_client(self.cosmos_database_name)
            self.container_proxy_async = db.get_container_client(self.stac_container_name)
        return self.container_proxy_async

    def _get_run_info(
        self, op_name: str, op_run_hash: str, container: ContainerProxy
    ) -> Optional[RunInfo]:
        try:
            retrieved_item = cast(Dict[str, Any], container.read_item(op_run_hash, op_name))
        except CosmosResourceNotFoundError:
            return None
        run_info_fields = [f.name for f in fields(RunInfo)]
        run_info_dict = {k: v for k, v in retrieved_item.items() if k in run_info_fields}
        return RunInfo(**run_info_dict)

    async def _get_run_info_async(
        self, op_name: str, op_run_hash: str, container: AsyncContainerProxy
    ) -> Optional[RunInfo]:
        try:
            retrieved_item = await container.read_item(op_run_hash, op_name)
        except CosmosResourceNotFoundError:
            return None
        run_info_fields = [f.name for f in fields(RunInfo)]
        run_info_dict = {k: v for k, v in retrieved_item.items() if k in run_info_fields}
        return RunInfo(**run_info_dict)

    def process_items(self, run_info: RunInfo, retrieved_items: List[Dict[str, Any]]):
        item_list_fields = [f.name for f in fields(ItemList)]
        items_dict: Dict[str, List[Dict[str, Any]]] = {}
        for i in retrieved_items:
            items_info_dict = {k: v for k, v in i.items() if k in item_list_fields}
            items_list = ItemList(**items_info_dict)
            output_name = items_list.output_name
            dict_list = items_dict.get(output_name, [])
            dict_list += items_list.items
            items_dict[output_name] = dict_list

        singular_input = run_info.singular_items
        retrieved_stac: ItemDict = {}

        for output_name, output_values in items_dict.items():
            converted_items = [Item.from_dict(ov, preserve_dict=False) for ov in output_values]
            if output_name in singular_input:
                retrieved_stac[output_name] = converted_items[0]
            else:
                retrieved_stac[output_name] = converted_items
        return retrieved_stac

    def _retrieve_items(self, run_info: RunInfo, container: ContainerProxy):
        retrieved_items = [container.read_item(i, run_info.op_name) for i in run_info.items]
        return self.process_items(run_info, retrieved_items)

    async def _retrieve_items_async(self, run_info: RunInfo, container: AsyncContainerProxy):
        retrieved_items = [await container.read_item(i, run_info.op_name) for i in run_info.items]
        return self.process_items(run_info, retrieved_items)

    def retrieve_output_from_input_if_exists(self, cache_info: CacheInfo) -> Optional[ItemDict]:
        container = self._get_container()
        run_info = self._get_run_info(cache_info.name, cache_info.hash, container)
        if run_info is None:
            return None

        return self._retrieve_items(run_info, container)

    async def retrieve_output_from_input_if_exists_async(
        self, cache_info: CacheInfo, **kwargs: Any
    ) -> Optional[ItemDict]:
        container = self._get_container_async()

        run_info = await self._get_run_info_async(cache_info.name, cache_info.hash, container)
        if run_info is None:
            return None

        return await self._retrieve_items_async(run_info, container)

    def remove(self, op_run_id: OpRunId):
        container = self._get_container()
        run_info = self._get_run_info(op_run_id.name, op_run_id.hash, container)
        if run_info is None:
            return None

        for i in run_info.items:
            try:
                container.delete_item(i, run_info.op_name)
            except CosmosResourceNotFoundError as er:
                self.logger.warning(
                    f"The item {i} that is a part of {op_run_id} does not exist in the "
                    f"Cosmos DB container: {er}"
                )

        try:
            container.delete_item(op_run_id.hash, op_run_id.name)
        except CosmosResourceNotFoundError as er:
            self.logger.warning(
                f"The item {op_run_id} does not exist in the Cosmos DB container: {er}"
            )


# Having to manually create Cosmos configuration so we can retrieve its
# key using a secret provider.
@dataclass
class CosmosStorageConfig(StorageConfig):
    _target_: str = "vibe_agent.storage.remote_storage.CosmosStorage"
    key: Any = MISSING
    asset_manager: BlobAssetManagerConfig = MISSING
    stac_container_name: Any = MISSING
    cosmos_database_name: Any = MISSING
    cosmos_url: Any = MISSING
