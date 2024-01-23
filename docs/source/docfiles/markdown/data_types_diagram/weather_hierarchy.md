<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class BaseVibe {
  }
  class DataVibe {
  }
  class Raster {
  }
  class GfsForecast {
  }
  class Grib {
  }
  class WeatherVibe {
  }
  DataVibe --|> BaseVibe
  Raster --|> DataVibe
  GfsForecast --|> DataVibe
  Grib --|> Raster
  WeatherVibe --|> DataVibe


</div>