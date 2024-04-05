<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">

classDiagram
  class BaseVibe {
  }
  class DataVibe {
  }
  class ADMAgPrescription {
  }
  class ADMAgPrescriptionInput {
  }
  class ADMAgPrescriptionMapInput {
  }
  class ADMAgSeasonalFieldInput {
  }
  class FertilizerInformation {
  }
  class HarvestInformation {
  }
  class OrganicAmendmentInformation {
  }
  class SeasonalFieldInformation {
  }
  class TillageInformation {
  }
  DataVibe --|> BaseVibe
  ADMAgPrescription --|> BaseVibe
  ADMAgPrescriptionInput --|> BaseVibe
  ADMAgPrescriptionMapInput --|> BaseVibe
  ADMAgSeasonalFieldInput --|> BaseVibe
  SeasonalFieldInformation --|> DataVibe


</div>