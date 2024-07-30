import os
from typing import Any, List

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from vibe_notebook.deepmc.utils import transform_to_array


def inference_deepmc(model_path: str, data_x: NDArray[Any], inference_hours: int):
    list_data_x = []
    for pred_idx in range(inference_hours):
        model_onnx_path = os.path.join(model_path, f"model_{pred_idx}", "export.onnx")
        session = onnxruntime.InferenceSession(model_onnx_path, None)
        data_in = {
            out.name: data_x[i].astype(np.float32) for i, out in enumerate(session.get_inputs())
        }

        result = session.run(None, input_feed=data_in)[0]
        result = result.astype(np.float32)
        result = transform_to_array(result, inference_hours)
        result = result[..., 0]
        list_data_x.append(result)
    return list_data_x


def inference_deepmc_post(
    model_path: str,
    post_data_x: List[NDArray[Any]],
):
    # Train Post-Processing Scaling Models
    inshape = len(post_data_x)
    mix_data_yhat = np.empty([post_data_x[0].shape[0], inshape, inshape])
    idx = 0

    for pred_idx, train_yhat in enumerate(post_data_x):
        post_model_onnx_path = os.path.join(model_path, f"model_{pred_idx}", "post", "export.onnx")
        post_session = onnxruntime.InferenceSession(post_model_onnx_path, None)
        data_in = {
            out.name: train_yhat.astype(np.float32)
            for i, out in enumerate(post_session.get_inputs())
        }
        result = post_session.run(None, input_feed=data_in)[0]
        mix_data_yhat[:, :, idx] = result
        idx = idx + 1
    return mix_data_yhat
