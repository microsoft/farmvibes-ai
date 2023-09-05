<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class BaseVibe {
  }
  class DataVibe {
  }
  class CDLProduct {
  }
  class ChirpsProduct {
  }
  class ClimatologyLabProduct {
  }
  class DemProduct {
  }
  class Era5Product {
  }
  class EsriLandUseLandCoverProduct {
  }
  class GEDIProduct {
  }
  class GNATSGOProduct {
  }
  class LandsatProduct {
  }
  class ModisProduct {
  }
  class NaipProduct {
  }
  DataVibe --|> BaseVibe
  CDLProduct --|> DataVibe
  ChirpsProduct --|> DataVibe
  ClimatologyLabProduct --|> DataVibe
  DemProduct --|> DataVibe
  Era5Product --|> DataVibe
  EsriLandUseLandCoverProduct --|> DataVibe
  GEDIProduct --|> DataVibe
  GNATSGOProduct --|> DataVibe
  LandsatProduct --|> DataVibe
  ModisProduct --|> DataVibe
  NaipProduct --|> DataVibe


</div>