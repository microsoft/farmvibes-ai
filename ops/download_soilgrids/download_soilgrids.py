import logging
import mimetypes
import os
import time
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, Final, List, cast

from owslib.wcs import WebCoverageService

from vibe_core.data import AssetVibe, DataVibe, Raster
from vibe_core.data.core_types import gen_guid, gen_hash_id

LOGGER = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_WAIT_S = 10


class SoilGridsWS:
    MAPS: Final[Dict[str, List[str]]] = {
        "wrb": [
            "World Reference Base classes and probabilites",
            "https://maps.isric.org/mapserv?map=/map/wrb.map",
        ],
        "bdod": ["Bulk density", "https://maps.isric.org/mapserv?map=/map/bdod.map"],
        "cec": [
            "Cation exchange capacity at ph 7",
            "https://maps.isric.org/mapserv?map=/map/cec.map",
        ],
        "cfvo": ["Coarse fragments volumetric", "https://maps.isric.org/mapserv?map=/map/cfvo.map"],
        "clay": ["Clay content", "https://maps.isric.org/mapserv?map=/map/clay.map"],
        "nitrogen": ["Nitrogen", "https://maps.isric.org/mapserv?map=/map/nitrogen.map"],
        "phh2o": ["Soil pH in H2O", "https://maps.isric.org/mapserv?map=/map/phh2o.map"],
        "sand": ["Sand content", "https://maps.isric.org/mapserv?map=/map/sand.map"],
        "silt": ["Silt content", "https://maps.isric.org/mapserv?map=/map/silt.map"],
        "soc": ["Soil organic carbon content", "https://maps.isric.org/mapserv?map=/map/soc.map"],
        "ocs": ["Soil organic carbon stock", "https://maps.isric.org/mapserv?map=/map/ocs.map"],
        "ocd": ["Organic carbon densities", "https://maps.isric.org/mapserv?map=/map/ocd.map"],
    }

    def __init__(self, map: str):
        self.map = map
        try:
            _, self.url = self.MAPS[map]
        except KeyError:
            raise ValueError(
                f"Map {map} cannot be found. "
                f"The maps available are: all {' '.join(self.MAPS.keys())}."
            )
        for retry in range(MAX_RETRIES):
            try:
                self.wcs = WebCoverageService(self.url, version="2.0.1")  # type: ignore
                return
            except Exception as e:
                LOGGER.warning(
                    f"Exception {e} requesting from {self.url}."
                    f" Retrying after {RETRY_WAIT_S}s ({retry+1}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_WAIT_S)
        raise RuntimeError(f"Failed request to {self.url} after {MAX_RETRIES} retries.")

    def get_ids(self) -> List[str]:
        return list(self.wcs.contents)  # type: ignore

    def download_id(self, id: str, tmpdir: str, input_item: DataVibe) -> Raster:
        if id not in self.get_ids():
            raise ValueError(
                f"Identifier {id} not found in {self.url}. Identifiers available"
                f" are: {' '.join(self.get_ids())}"
            )
        bbox = input_item.bbox
        subsets = [("long", bbox[0], bbox[2]), ("lat", bbox[1], bbox[3])]
        for retry in range(MAX_RETRIES):
            try:
                response = self.wcs.getCoverage(  # type: ignore
                    identifier=[id],
                    subsets=subsets,
                    SUBSETTINGCRS="http://www.opengis.net/def/crs/EPSG/0/4326",
                    OUTPUTCRS="http://www.opengis.net/def/crs/EPSG/0/4326",
                    format="image/tiff",
                )
                fpath = os.path.join(tmpdir, f"{id}_{gen_guid()}.tif")
                with open(fpath, "wb") as file:
                    file.write(response.read())
                vibe_asset = AssetVibe(
                    reference=fpath, type=cast(str, mimetypes.guess_type(fpath)[0]), id=gen_guid()
                )
                res = Raster(
                    id=gen_hash_id(
                        f"soilgrids_{self.map}_{id}",
                        input_item.geometry,
                        (datetime(2022, 1, 1), datetime(2022, 1, 1)),  # dummy date
                    ),
                    time_range=input_item.time_range,
                    geometry=input_item.geometry,
                    assets=[vibe_asset],
                    bands={f"{self.map}:{id}": 0},
                )
                return res
            except Exception as e:
                LOGGER.warning(
                    f"Exception {e} downloading {id} from {self.url}."
                    f" Retrying after {RETRY_WAIT_S}s ({retry+1}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_WAIT_S)
        raise RuntimeError(f"Failed request for {id} in {self.url} after {MAX_RETRIES} retries.")


class CallbackBuilder:
    def __init__(self, map: str, identifier: str):
        self.tmp_dir = TemporaryDirectory()
        self.map = map
        self.identifier = identifier

    def __call__(self):
        def download_soilgrids(
            input_item: DataVibe,
        ) -> Dict[str, Raster]:
            sg = SoilGridsWS(self.map)
            res = sg.download_id(self.identifier, self.tmp_dir.name, input_item)
            return {"downloaded_raster": res}

        return download_soilgrids

    def __del__(self):
        self.tmp_dir.cleanup()
