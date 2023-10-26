<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class AssetVibe {
  }
  class BaseVibe {
  }
  class CarbonOffsetInfo {
  }
  class DataSequence {
  }
  class DataSummaryStatistics {
  }
  class DataVibe {
  }
  class ExternalReference {
  }
  class ExternalReferenceList {
  }
  class FoodFeatures {
  }
  class FoodVibe {
  }
  class GHGFlux {
  }
  class GHGProtocolVibe {
  }
  class GeometryCollection {
  }
  class ProteinSequence {
  }
  class PydanticAssetVibe {
  }
  class TimeSeries {
  }
  class Tmp {
  }
  class TypeDictVibe {
  }
  class TypeParser {
  }
  class UnresolvedDataVibe {
  }
  PydanticAssetVibe --|> AssetVibe
  CarbonOffsetInfo --|> DataVibe
  DataSequence --|> DataVibe
  DataSummaryStatistics --|> DataVibe
  DataVibe --|> BaseVibe
  ExternalReference --|> DataVibe
  ExternalReferenceList --|> DataVibe
  FoodFeatures --|> DataVibe
  FoodVibe --|> BaseVibe
  GHGFlux --|> DataVibe
  GHGProtocolVibe --|> DataVibe
  GeometryCollection --|> DataVibe
  ProteinSequence --|> DataVibe
  TimeSeries --|> DataVibe
  UnresolvedDataVibe --|> BaseVibe


</div>