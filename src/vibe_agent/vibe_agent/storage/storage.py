"""
Storage module for TerraVibes. Helps store, index, retrieve, and catalog geospatial knowledge that
an instance of TerraVibes contains.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from hydra_zen import builds
from pystac.asset import Asset

from vibe_common.schemas import CacheInfo, ItemDict, OpRunId
from vibe_core.utils import ensure_list

from .asset_management import AssetManager, AssetManagerConfig


class AssetCopyHandler:
    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager

    def _copy_asset(self, guid: str, asset: Asset):
        file_path = asset.get_absolute_href()
        assert file_path is not None
        asset.href = self.asset_manager.store(guid, file_path)

    def _copy_prepared_assets(self, assets_to_copy: Dict[str, Asset]):
        copied_assets: List[str] = []
        try:
            for guid, asset in assets_to_copy.items():
                self._copy_asset(guid, asset)
                copied_assets.append(guid)
        except Exception:
            for f in copied_assets:
                self.asset_manager.remove(f)
            raise

    def _prepare_assets(self, items: ItemDict):
        assets: Dict[str, Asset] = {}
        for item in items.values():
            item = ensure_list(item)
            for i in item:
                assets.update(i.assets)
        return assets

    def copy_assets(self, items: ItemDict):
        assets = self._prepare_assets(items)
        self._copy_prepared_assets(assets)

        return items


class Storage(ABC):
    """
    The TerraVibes storage class contains abstract methods that have to be implemented. The abstract
    methods are "store", "retrieve", and "retrieve_output_from_input_if_exists". Store and retrieve
    are self explanatory. The latter one helps  retrieve data by querying with the inputs that
    generated the output that the user is looking for. These methods are mandatory when
    implementing a storage class in TerraVibes.
    """

    asset_manager: AssetManager
    asset_copy_handler: AssetCopyHandler
    HASH_FIELD: str = "vibe_op_hash"

    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager
        self.asset_handler = AssetCopyHandler(asset_manager)

    @abstractmethod
    def store(self, run_id: str, items: ItemDict, cache_info: CacheInfo) -> ItemDict:
        raise NotImplementedError

    def retrieve(self, input_items: ItemDict) -> ItemDict:
        """
        Method to retrieve a list of items from the current TerraVibes storage STAC catalog
        """
        for possible_item_list in input_items.values():
            items = ensure_list(possible_item_list)
            for item in items:
                for guid, asset in item.assets.items():
                    asset.href = self.asset_manager.retrieve(guid)

        return input_items

    @abstractmethod
    def retrieve_output_from_input_if_exists(self, cache_info: CacheInfo) -> Optional[ItemDict]:
        raise NotImplementedError

    @abstractmethod
    async def retrieve_output_from_input_if_exists_async(
        self, cache_info: CacheInfo, **kwargs: Any
    ) -> Optional[ItemDict]:
        raise NotImplementedError

    @abstractmethod
    def remove(self, op_run_id: OpRunId):
        """
        Method to delete a STAC catalog from storage. Note: this does not remove the assets
        referenced by a STAC catalog.
        """
        raise NotImplementedError


StorageConfig = builds(
    Storage,
    asset_manager=AssetManagerConfig,
    zen_dataclass={
        "module": "vibe_agent.storage.storage",
        "cls_name": "StorageConfig",
    },
)
