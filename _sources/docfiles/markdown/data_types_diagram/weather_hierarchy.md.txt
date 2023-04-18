<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class BaseVibe {
  }
  class DataVibe {
  }
  class GfsForecast {
  }
  class WeatherVibe {
  }
  DataVibe --|> BaseVibe
  GfsForecast --|> DataVibe
  WeatherVibe --|> DataVibe


</div>