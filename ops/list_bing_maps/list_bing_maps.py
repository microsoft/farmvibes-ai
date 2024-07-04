import hashlib
from datetime import datetime
from typing import Dict, List, Optional

from pystac.item import Item

from vibe_core.data import DataVibe
from vibe_core.data.products import BingMapsProduct
from vibe_lib.bing_maps import MAX_ZOOM_LEVEL, MIN_ZOOM_LEVEL, BingMapsCollection


class CallbackBuilder:
    def __init__(
        self,
        api_key: str,
        zoom_level: int,
        imagery_set: str,
        map_layer: str,
        orientation: Optional[float],
    ):
        if not api_key:
            raise ValueError("BingMaps API key was not provided.")
        if imagery_set != "Aerial":
            raise ValueError("Only imagery set 'Aerial' is supported.")
        if map_layer != "Basemap":
            raise ValueError("Only map layer 'Basemap' is supported.")
        if orientation is not None:
            raise ValueError("Setting an orientation is currently not supported.")
        if zoom_level < MIN_ZOOM_LEVEL or zoom_level > MAX_ZOOM_LEVEL:
            raise ValueError(
                f"Zoom level must be within [{MIN_ZOOM_LEVEL}, {MAX_ZOOM_LEVEL}]. "
                f"Found {zoom_level}."
            )

        self.collection = BingMapsCollection(api_key)
        self.zoom_level = zoom_level
        self.imagery_set = imagery_set
        self.map_layer = map_layer
        self.orientation = 0.0 if orientation is None else orientation

    def convert_product(self, item: Item) -> BingMapsProduct:
        assert item.geometry is not None, "input Item has no geometry"

        product = BingMapsProduct(
            id=hashlib.sha256(
                (f"bingmaps-{item.id}-{self.imagery_set}-{self.map_layer}").encode()
            ).hexdigest(),
            time_range=(datetime.now(), datetime.now()),
            geometry=item.geometry,
            assets=[],
            url=item.properties["url"],
            zoom_level=self.zoom_level,
            imagery_set=self.imagery_set,
            map_layer=self.map_layer,
            orientation=self.orientation,
        )
        return product

    def __call__(self):
        def list_bing_maps(
            user_input: DataVibe,
        ) -> Dict[str, List[BingMapsProduct]]:
            items = self.collection.query_tiles(user_input.bbox, self.zoom_level)

            if not items:
                raise RuntimeError("No products found for input geometry and zoom level.")

            products = [self.convert_product(item) for item in items]
            return {"products": products}

        return list_bing_maps
