# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from azure.core.credentials import TokenCredential
from pystac import Asset, Item

from vibe_common.messaging import OpIOType
from vibe_common.tokens import BlobTokenManagerConnectionString, BlobTokenManagerCredentialed
from vibe_core.data.utils import deserialize_stac, serialize_stac
from vibe_core.datamodel import RunConfigUser
from vibe_core.utils import ensure_list


class HrefHandler(ABC):
    @abstractmethod
    def _update_asset(self, asset: Asset):
        raise NotImplementedError

    def _parse_item(self, item: Item):
        assets = item.get_assets()
        for asset in assets:
            self._update_asset(assets[asset])
        return item

    def _parse_items(self, obj: Union[Item, List[Item]]) -> Union[Item, List[Item]]:
        if isinstance(obj, Item):
            return self._parse_item(obj)
        else:
            return [self._parse_item(item) for item in obj]

    def _run(self, out: OpIOType) -> OpIOType:
        result = {}
        for key in out:
            items = deserialize_stac(out[key])
            items = ensure_list(items)
            for item in items:
                item.clear_links()
            result[key] = serialize_stac(self._parse_items(items))
        return result

    def handle(self, original_response: RunConfigUser) -> RunConfigUser:
        original_response.output = self._run(original_response.output)
        return original_response


class LocalHrefHandler(HrefHandler):
    def __init__(self, assets_dir: Union["str", Path]):
        super().__init__()
        self.assets_dir = assets_dir if isinstance(assets_dir, Path) else Path(assets_dir)

    def _update_asset(self, asset: Asset):
        asset_href_path = Path(asset.href).resolve()
        parent_name = asset_href_path.parent.name
        asset_name = asset_href_path.name

        asset.href = str(self.assets_dir / Path(parent_name) / asset_name)


class BlobHrefHandler(HrefHandler):
    def __init__(
        self, credential: Optional[TokenCredential] = None, connection_string: Optional[str] = None
    ):
        super().__init__()
        if connection_string is not None:
            self.manager = BlobTokenManagerConnectionString(connection_string=connection_string)
        else:
            self.manager = BlobTokenManagerCredentialed(credential=credential)

    def _update_asset(self, asset: Asset):
        asset.href = self.manager.sign_url(asset.href)
