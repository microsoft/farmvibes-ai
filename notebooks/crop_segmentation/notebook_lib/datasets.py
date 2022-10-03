import json
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import rasterio
import rasterio.merge
import torch
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rtree.index import Index, Property
from torch import Tensor
from torchgeo.datasets import BoundingBox, RasterDataset
from vibe_core.data import Raster
from vibe_core.data.rasters import CategoricalRaster


class NDVIDataset(RasterDataset):
    #: Color map for the dataset, used for plotting
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(
        self,
        ndvi_rasters: List[Raster],
        stack_n_bands: int = 37,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            ndvi_rasters: list of Rasters output by TerraVibes workflow
            stack_n_bands: number of bands of the ndvi stack (available
                rasters will be temporally sampled to compose a stack
                with this number of bands)
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        if stack_n_bands > len(ndvi_rasters):
            raise ValueError(
                f"Number of NDVI rasters must be >= stack_n_bands, found {len(ndvi_rasters)}"
            )

        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        self.index_size = 0

        # Sort raster by date
        self.ndvi_rasters = sorted(ndvi_rasters, key=lambda x: x.time_range[0])
        self.stack_n_bands = stack_n_bands
        self.cache = cache

        # Read color map
        vis_asset = self.ndvi_rasters[0].visualization_asset
        with open(vis_asset.local_path) as mtdt:
            metadata = json.load(mtdt)
            self.cmap = metadata["colormap"]

        # Build the index, temporally sampling rasters
        for year, grouped_rasters in groupby(self.ndvi_rasters, lambda x: x.time_range[0].year):
            # Group rasters by year and find unique dates (might have multiple rasters for a date)
            rasters = list(grouped_rasters)
            unique_dates = set([raster.time_range[0] for raster in rasters])
            n_unique_dates = len(unique_dates)

            # Raise exception if we cannot build a stack
            if n_unique_dates < self.stack_n_bands:
                raise ValueError(
                    f"{n_unique_dates} unique dates for {year}, "
                    f"expected at least {self.stack_n_bands}"
                )

            # Define sampling interval for dates
            selected_date_idxs = np.round(
                np.linspace(0, n_unique_dates - 1, self.stack_n_bands)
            ).astype(int)
            selected_rasters = [rasters[idx] for idx in selected_date_idxs]

            # Loop through the selected rasters
            for raster in selected_rasters:
                try:
                    self._add_raster_to_index(raster)
                except RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue

        if self.index_size == 0:
            raise FileNotFoundError(
                f"Couldn't read {self.__class__.__name__} data from ndvi_rasters"
            )

    def _add_raster_to_index(self, raster: Raster):
        filepath = raster.raster_asset.local_path

        with rasterio.open(filepath) as src:
            crs = src.crs
            res = src.res[0]

            with WarpedVRT(src, crs=crs) as vrt:
                minx, miny, maxx, maxy = vrt.bounds

        start_date, end_date = raster.time_range
        coords = (
            minx,
            maxx,
            miny,
            maxy,
            start_date.timestamp(),
            end_date.timestamp(),
        )
        self.index.insert(self.index_size, coords, filepath)
        self.index_size += 1

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

        hit_samples = [hit for hit in self.index.intersection(tuple(query), objects=True)]

        if not hit_samples:
            raise IndexError(f"query: {query} not found in index with bounds: {self.bounds}")

        filepaths: List[str] = [hit.object for hit in hit_samples]  # type:ignore
        maxt_timestamp = [hit.bounds[-1] for hit in hit_samples]  # type:ignore

        data_list: List[Tensor] = []
        spatial_merge_list: List[str] = []
        merge_timestamp = maxt_timestamp[0]
        for filepath, ts in zip(filepaths, maxt_timestamp):
            # if date matches the merge_timestamp, add the raster to be merged
            if ts == merge_timestamp:
                spatial_merge_list.append(filepath)
                merge_timestamp = ts
            else:  # date changed, merge rasters and add new raster to the list
                data_list.append(self._spatial_merge_files(spatial_merge_list, query))
                spatial_merge_list = [filepath]
                merge_timestamp = ts

        # merge the remaining rasters
        data_list.append(self._spatial_merge_files(spatial_merge_list, query))

        # Stack ndvi rasters in the channel dimension
        data = torch.cat(data_list, dim=0)

        sample = {"image": data, "crs": self.crs, "bbox": query}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _spatial_merge_files(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load and spatially merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = int(round((query.maxx - query.minx) / self.res))
            out_height = int(round((query.maxy - query.miny) / self.res))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(out_shape=out_shape, window=from_bounds(*bounds, src.transform))
        else:
            dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res)

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor


class CDLMask(RasterDataset):
    """
    Binary mask dataset based on the choice of a CDL index subset to serve as a positive indices.
    """

    is_image = False

    def __init__(
        self,
        cdl_rasters: List[CategoricalRaster],
        positive_indices: Sequence[int],
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ):
        """Initialize a new Dataset instance.

        Args:
            cdl_rasters: list of Rasters output by TerraVibes workflow
            positive_indices: crop indices to consider as the positive label
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.positive_indices = torch.as_tensor(positive_indices)
        self.transforms = transforms
        self.cdl_rasters = sorted(cdl_rasters, key=lambda x: x.time_range[0])
        self.cache = cache

        # Read color map
        vis_asset = self.cdl_rasters[0].visualization_asset
        with open(vis_asset.local_path) as mtdt:
            metadata = json.load(mtdt)
            self.cmap = metadata["colormap"]

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Populate the dataset index
        sample_idx = 0
        for raster in self.cdl_rasters:
            filepath = raster.raster_asset.local_path
            try:
                with rasterio.open(filepath) as src:
                    crs = src.crs
                    res = src.res[0]

                    with WarpedVRT(src, crs=crs) as vrt:
                        minx, miny, maxx, maxy = vrt.bounds
            except RasterioIOError:
                # Skip files that rasterio is unable to read
                continue
            else:
                start_date, end_date = raster.time_range
                coords = (
                    minx,
                    maxx,
                    miny,
                    maxy,
                    start_date.timestamp(),
                    end_date.timestamp(),
                )
                self.index.insert(sample_idx, coords, filepath)
                sample_idx += 1

        if sample_idx == 0:
            raise FileNotFoundError(
                f"Couldn't read {self.__class__.__name__} data from ndvi_rasters"
            )

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        sample = super().__getitem__(query)
        sample["mask"] = torch.isin(sample["mask"], self.positive_indices).to(torch.float32)
        return sample
