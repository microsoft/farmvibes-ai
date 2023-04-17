<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class AirbusPrice {
  }
  class AirbusProduct {
  }
  class AirbusRaster {
  }
  class BaseVibe {
  }
  class DataVibe {
  }
  class Raster {
  }
  AirbusPrice --|> DataVibe
  AirbusProduct --|> DataVibe
  AirbusRaster --|> AirbusProduct
  AirbusRaster --|> Raster
  DataVibe --|> BaseVibe
  Raster --|> DataVibe


</div>