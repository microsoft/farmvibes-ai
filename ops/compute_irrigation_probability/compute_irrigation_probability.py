# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vibe_core.data import AssetVibe, LandsatRaster, Raster, gen_guid
from vibe_lib.raster import load_raster_match


# Define a function for ngi, egi, and lst data treatment
def preprocess_raster_values(raster: xr.DataArray):
    raster_values = raster.values.ravel()

    # Handle NaN and Inf values
    raster_values[np.isnan(raster_values)] = -9999
    raster_values[np.isinf(raster_values)] = -9999

    # Replace -9999 with 0
    raster_values = np.where(raster_values == -9999, 0, raster_values)

    return raster_values


class CallbackBuilder:
    def __init__(self, coef_ngi: float, coef_egi: float, coef_lst: float, intercept: float):
        # Create temporary directory to store our new data, which will be transfered to our storage
        # automatically when the op is run in a workflow
        self.tmp_dir = TemporaryDirectory()

        # Set Parameters
        self.coef_ngi = coef_ngi
        self.coef_egi = coef_egi
        self.coef_lst = coef_lst
        self.intercept = intercept

    def __call__(self):
        def callback(
            landsat_raster: LandsatRaster,
            ngi: Raster,
            egi: Raster,
            lst: Raster,
            cloud_water_mask_raster: Raster,
        ) -> Dict[str, Raster]:
            # Get cloud water mask layer
            cloud_water_mask = load_raster_match(cloud_water_mask_raster, landsat_raster)[0]

            # Get ngi, egi, and lst layers
            ngi1 = load_raster_match(ngi, landsat_raster)[0]
            egi1 = load_raster_match(egi, landsat_raster)[0]
            lst1 = load_raster_match(lst, landsat_raster)[0]

            ngi_values = preprocess_raster_values(ngi1)
            egi_values = preprocess_raster_values(egi1)
            lst_values = preprocess_raster_values(lst1)

            # Reduce dimension
            x = np.stack((ngi_values, egi_values, lst_values), axis=1)
            x = x.astype(float)

            # Apply scaler
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            # Create a logistic regression model
            model = LogisticRegression()

            # Set the coefficients and intercept
            coef_ = np.array([[self.coef_ngi, self.coef_ngi, self.coef_lst]])
            intercept_ = [self.intercept]
            classes_ = np.array(["1", "2"])

            # Assign the coefficients and intercept to the model
            model.coef_ = coef_
            model.intercept_ = intercept_
            model.classes_ = classes_

            # Make predictions using the model
            predicted_labels = model.predict_proba(x_scaled)[:, 0]

            # Assign shape
            predicted_labels = predicted_labels.reshape(cloud_water_mask.shape)

            # Treat the result with cloud water mask
            predicted_labels = predicted_labels * cloud_water_mask

            # Create a new DataArray with predicted_labels and the same dimensions as ngi
            predicted_labels_xr = xr.DataArray(
                predicted_labels,
                dims=cloud_water_mask.dims,
                coords=cloud_water_mask.coords,
            )

            # Save the DataArray to a raster file
            filepath = os.path.join(self.tmp_dir.name, "irrigation_probability.tif")
            predicted_labels_xr.rio.to_raster(filepath)
            irr_prob_asset = AssetVibe(reference=filepath, type="image/tiff", id=gen_guid())
            return {
                "irrigation_probability": Raster.clone_from(
                    landsat_raster,
                    id=gen_guid(),
                    assets=[irr_prob_asset],
                    bands={"irrigation_probability": 0},
                )
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
