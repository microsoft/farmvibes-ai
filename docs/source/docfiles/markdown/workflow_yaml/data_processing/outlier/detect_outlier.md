# data_processing/outlier/detect_outlier

```yaml

name: detect_outlier
sources:
  rasters:
  - outlier.rasters
sinks:
  segmentation: outlier.segmentation
  heatmap: outlier.heatmap
  outliers: outlier.outliers
  mixture_means: outlier.mixture_means
parameters:
  threshold: null
tasks:
  outlier:
    op: detect_outliers
    parameters:
      threshold: '@from(threshold)'
edges: null
description:
  short_description: Fits a single-component Gaussian Mixture Model (GMM) over input
    data to detect outliers according to the threshold parameter.
  long_description: The workflow outputs segmentation and outlier maps based on the
    threshold parameter and the likelihood of each sample belonging to the GMM component.
    It also yields heatmaps of the likelihood, and the mean of GMM's component.
  sources:
    rasters: Input rasters.
  sinks:
    segmentation: Segmentation maps based on the likelihood of each sample belonging
      to the GMM's single-component.
    heatmap: Likelihood maps.
    outliers: Outlier maps based on the thresholded likelihood map.
    mixture_means: Mean of the GMM.
  parameters:
    threshold: Likelihood threshold value to consider a sample as an outlier.


```

```{mermaid}
    graph TD
    inp1>rasters]
    out1>segmentation]
    out2>heatmap]
    out3>outliers]
    out4>mixture_means]
    tsk1{{outlier}}
    inp1>rasters] -- rasters --> tsk1{{outlier}}
    tsk1{{outlier}} -- segmentation --> out1>segmentation]
    tsk1{{outlier}} -- heatmap --> out2>heatmap]
    tsk1{{outlier}} -- outliers --> out3>outliers]
    tsk1{{outlier}} -- mixture_means --> out4>mixture_means]
```