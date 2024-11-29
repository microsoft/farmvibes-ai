# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import logging
import os
import shutil
from concurrent.futures import Executor
from typing import Any, Dict, List, Optional, Union, cast

from hydra_zen import MISSING, builds
from pystac.catalog import Catalog, CatalogType
from pystac.collection import Collection, Extent
from pystac.item import Item
from pystac.stac_io import DefaultStacIO

from vibe_common.schemas import CacheInfo, OpRunId
from vibe_core.utils import ensure_list

from .asset_management import LocalFileAssetManagerConfig
from .storage import AssetManager, ItemDict, Storage, StorageConfig


class LocalStacIO(DefaultStacIO):
    def stac_object_from_dict(
        self,
        d: Dict[str, Any],
        href: Optional[str] = None,
        root: Optional[Catalog] = None,
        preserve_dict: bool = False,
    ) -> Any:
        return super().stac_object_from_dict(d, href, root, False)


class LocalResourceExistsError(RuntimeError):
    pass


class LocalStorage(Storage):
    """
    This class implements the Storage abstract class.
    """

    IS_SINGULAR_FIELD = "terravibe_is_singular"
    COLLECTION_TYPE = CatalogType.SELF_CONTAINED
    CATALOG_TYPE = CatalogType.RELATIVE_PUBLISHED

    def __init__(self, local_path: str, asset_manager: AssetManager):
        """
        Initializer expects a directory path where catalogs can be stored
        """
        super().__init__(asset_manager)
        self.path = local_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stac_io = LocalStacIO()

    def _retrieve_items(self, catalog: Catalog) -> ItemDict:
        output: ItemDict = {}
        for c in catalog.get_collections():
            output[c.id] = list(c.get_items())
            if c.extra_fields[self.IS_SINGULAR_FIELD]:  # type: ignore
                output[c.id] = cast(List[Item], output[c.id])[0]
        return output

    def _create_output_collection(
        self, output_name: str, items: Union[Item, List[Item]]
    ) -> Collection:
        extra_info: Dict[str, bool] = {self.IS_SINGULAR_FIELD: not isinstance(items, list)}
        output_items = ensure_list(items)
        extent = Extent.from_items(output_items)
        description = f"Stores op output {output_name} for a unique op run."
        output_collection = Collection(
            id=output_name,
            description=description,
            extent=extent,
            catalog_type=self.COLLECTION_TYPE,
            extra_fields=extra_info,
        )
        output_collection.add_items(output_items)

        return output_collection

    def retrieve_output_from_input_if_exists(
        self,
        cache_info: CacheInfo,
    ) -> Optional[ItemDict]:
        """
        Method to help users to skip computation if the result of the previous outputs from input
        and operator combo has been memo-ized as a catalog in the TerraVibes storage system
        """
        catalog_path = self.get_catalog_path(cache_info.hash, cache_info.name)
        if os.path.exists(catalog_path):
            catalog = Catalog.from_file(
                os.path.join(catalog_path, Catalog.DEFAULT_FILE_NAME), stac_io=self.stac_io
            )
            return self._retrieve_items(catalog)

        return None

    async def retrieve_output_from_input_if_exists_async(
        self, cache_info: CacheInfo, **kwargs: Any
    ):
        executor: Executor = cast(Executor, kwargs["executor"])
        return await asyncio.get_running_loop().run_in_executor(
            executor, self.retrieve_output_from_input_if_exists, cache_info
        )

    def create_run_collection(
        self,
        run_id: str,
        catalog_path: str,
        items: ItemDict,
        extra_info: Dict[str, Any],
    ) -> Catalog:
        description = f"Collection of outputs of run id {run_id}."
        run_catalog = Catalog(
            id=run_id,
            description=description,
            href=catalog_path,
            catalog_type=self.CATALOG_TYPE,
            extra_fields=extra_info,
        )
        for output_name, output_items in items.items():
            output_collection = self._create_output_collection(output_name, output_items)
            run_catalog.add_child(output_collection)

        return run_catalog

    def get_catalog_path(self, op_hash: str, op_name: str) -> str:
        """
        Each catalog has a directory and json file where the corresponding assets and files are
        stored/indexed
        """
        return os.path.join(self.path, op_name, op_hash)

    def _catalog_cleanup(self, catalog: Catalog):
        catalog_path = catalog.get_self_href()
        assert catalog_path is not None, f"Catalog {catalog.id} does not have an href."
        catalog.normalize_hrefs(catalog_path)
        catalog.make_all_asset_hrefs_relative()

    def store(self, run_id: str, items_to_store: ItemDict, cache_info: CacheInfo) -> ItemDict:
        """
        Method to store a given list of items to current TerraVibes storage STAC catalog
        This method must be atomic -- that is all of it happens or none of it happens
        This method must be consistent -- that is the assets/items referenced by catalogs must be in
          storage & vice-versa
        This method must be isolated -- applications should be able to call multiple store
          operations simultaneously and safely
        This method must be durable -- all changes must be available across crashes unless there
          is a catastrophic failure
        This method must be performant -- it should support 1000s/100s/10s of
          assets/catalogs/workflows being updated simultaneously
        """
        catalog_path = self.get_catalog_path(cache_info.hash, cache_info.name)
        items_to_store = self.asset_handler.copy_assets(items_to_store)
        catalog = self.create_run_collection(
            run_id, catalog_path, items_to_store, cache_info.as_storage_dict()
        )
        self._catalog_cleanup(catalog)
        if not os.path.exists(catalog_path):
            catalog.save(stac_io=self.stac_io)
        else:
            raise LocalResourceExistsError(
                f"Op output already exists in storage for {cache_info.name} with id {run_id}."
            )

        return items_to_store

    def remove(self, op_run_id: OpRunId):
        catalog_path = self.get_catalog_path(op_run_id.hash, op_run_id.name)

        if not os.path.exists(catalog_path):
            self.logger.info(
                f"Asked to remove nonexistent catalog with op name {op_run_id.name} and hash "
                f"{op_run_id.hash}."
            )
            return

        try:
            shutil.rmtree(catalog_path)
        except OSError:
            self.logger.exception(f"Error removing catalog for op run {op_run_id}.")
            raise


LocalStorageConfig = builds(
    LocalStorage,
    local_path=MISSING,
    asset_manager=LocalFileAssetManagerConfig(MISSING),
    builds_bases=(StorageConfig,),
    zen_dataclass={
        "module": "vibe_agent.storage.local_storage",
        "cls_name": "LocalStorageConfig",
    },
)
