<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class Enum {
  }
  class StrEnum {
  }
  class BaseVibe {
  }
  class DataSequence {
  }
  class DataVibe {
  }
  class CategoricalRaster {
  }
  class CloudRaster {
  }
  class Raster {
  }
  class RasterSequence {
  }
  class CloudMask {
  }
  class DownloadedSentinel1Product {
  }
  class DownloadedSentinel2Product {
  }
  class S2ProcessingLevel {
  }
  class Sentinel1Product {
  }
  class Sentinel1Raster {
  }
  class Sentinel1RasterOrbitGroup {
  }
  class Sentinel1RasterTileSequence {
  }
  class Sentinel2CloudMask {
  }
  class Sentinel2CloudMaskOrbitGroup {
  }
  class Sentinel2CloudMaskTileSequence {
  }
  class Sentinel2CloudProbability {
  }
  class Sentinel2Product {
  }
  class Sentinel2Raster {
  }
  class Sentinel2RasterOrbitGroup {
  }
  class Sentinel2RasterTileSequence {
  }
  class SentinelProduct {
  }
  class SentinelRaster {
  }
  class SpaceEyeRaster {
  }
  class SpaceEyeRasterSequence {
  }
  class TileSequence {
  }
  class TiledSentinel1Product {
  }
  StrEnum --|> Enum
  DataSequence --|> DataVibe
  DataVibe --|> BaseVibe
  CategoricalRaster --|> Raster
  CloudRaster --|> Raster
  Raster --|> DataVibe
  RasterSequence --|> DataSequence
  RasterSequence --|> Raster
  CloudMask --|> CategoricalRaster
  CloudMask --|> CloudRaster
  CloudMask --|> Sentinel2Product
  DownloadedSentinel1Product --|> Sentinel1Product
  DownloadedSentinel2Product --|> Sentinel2Product
  S2ProcessingLevel --|> StrEnum
  Sentinel1Product --|> SentinelProduct
  Sentinel1Raster --|> Raster
  Sentinel1Raster --|> Sentinel1Product
  Sentinel1RasterOrbitGroup --|> Sentinel1Raster
  Sentinel1RasterTileSequence --|> Sentinel1Raster
  Sentinel1RasterTileSequence --|> TileSequence
  Sentinel2CloudMask --|> CloudMask
  Sentinel2CloudMask --|> Sentinel2Product
  Sentinel2CloudMaskOrbitGroup --|> Sentinel2CloudMask
  Sentinel2CloudMaskTileSequence --|> Sentinel2CloudMask
  Sentinel2CloudMaskTileSequence --|> TileSequence
  Sentinel2CloudProbability --|> CloudRaster
  Sentinel2CloudProbability --|> Sentinel2Product
  Sentinel2Product --|> SentinelProduct
  Sentinel2Raster --|> Raster
  Sentinel2Raster --|> Sentinel2Product
  Sentinel2RasterOrbitGroup --|> Sentinel2Raster
  Sentinel2RasterTileSequence --|> Sentinel2Raster
  Sentinel2RasterTileSequence --|> TileSequence
  SentinelProduct --|> DataVibe
  SentinelRaster --|> Raster
  SentinelRaster --|> SentinelProduct
  SpaceEyeRaster --|> Sentinel2Raster
  SpaceEyeRasterSequence --|> SpaceEyeRaster
  SpaceEyeRasterSequence --|> TileSequence
  TileSequence --|> RasterSequence
  TiledSentinel1Product --|> DownloadedSentinel1Product


</div>