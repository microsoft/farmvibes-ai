"""Export SAM models to ONNX files and add them to FarmVibes.AI cluster.

This script downloads the SAM models from the Segment Anything project, exports them to
ONNX files, and adds them to the FarmVibes.AI cluster. The script was heavily inspired by
Visheratin's export_onnx_model script available in:
# https://github.com/visheratin/segment-anything/blob/main/scripts/export_onnx_model.py
"""

import argparse
import os
import subprocess
import warnings
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import onnx
import torch
from onnx.external_data_helper import convert_model_to_external_data
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam
from segment_anything.utils.onnx import SamOnnxModel

from vibe_core.file_downloader import download_file

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


@dataclass
class ModelInfo:
    url: str
    should_use_data_file: bool


MODELS = {
    "vit_b": ModelInfo(
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        should_use_data_file=False,
    ),
    "vit_l": ModelInfo(
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        should_use_data_file=False,
    ),
    "vit_h": ModelInfo(
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        should_use_data_file=True,
    ),
}
RETURN_SINGLE_MASK = True
ONNX_OPSET = 17

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download SAM model(s), export to ONNX files, and add to FarmVibes.AI cluster."
    )

    parser.add_argument(
        "--cluster",
        choices=["local", "remote"],
        default="local",
        help="The type of cluster to add the models to (local or remote).",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["vit_b", "vit_l", "vit_h"],
        required=True,
        help="A list of SAM model types to export (among 'vit_b', 'vit_l', and 'vit_h').",
    )

    return parser.parse_args()


def export_to_onnx(
    model: torch.nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    dynamic_axes: Dict[str, Dict[int, str]],
    output_names: List[str],
    output_path: str,
):
    """Export the model to ONNX file.

    Args:
        model: The model to export.
        dummy_inputs: A dictionary of dummy inputs to the model.
        dynamic_axes: A dictionary of dynamic axes for the model.
        output_names: A list of output names for the model.
        output_path: The path to save the exported ONNX file.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # type: ignore
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_path, "wb") as f:
            print(f"Exporting onnx model to {output_path}...")
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                f,  # type: ignore
                export_params=True,
                verbose=False,
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


def export_model(model_type: str, downloaded_path: str, dir_path: str) -> Tuple[str, str]:
    """Export the SAM model to ONNX files.

    Args:
        model_type: The SAM model type to export (vit_b, vit_l, vit_h).
        downloaded_path: The path to the downloaded SAM model.
        dir_path: The directory path to save the exported ONNX files.
    """
    encoder_output = os.path.join(dir_path, f"{model_type}_encoder.onnx")
    encoder_data_file = (
        os.path.join(dir_path, f"{model_type}_encoder_data_file.onnx")
        if MODELS[model_type].should_use_data_file
        else None
    )

    decoder_output = os.path.join(dir_path, f"{model_type}_decoder.onnx")

    sam = sam_model_registry[model_type](checkpoint=downloaded_path)

    encoder_path = export_encoder(sam, encoder_output, encoder_data_file)
    decoder_path = export_decoder(sam, decoder_output)

    return (encoder_path, decoder_path)


def export_encoder(sam: Sam, output: str, data_file_output: Optional[str]) -> str:
    """Export the encoder model to ONNX file.

    Args:
        sam: The SAM model.
        output: The path to save the exported ONNX file.
        data_file_output: The path to save the external data file.
    Returns:
        The path to the exported ONNX file.
    """
    dynamic_axes = {
        "x": {0: "batch"},
    }
    dummy_inputs = {
        "x": torch.randn(1, 3, 1024, 1024, dtype=torch.float),
    }
    _ = sam.image_encoder(**dummy_inputs)

    output_names = ["image_embeddings"]

    export_to_onnx(sam.image_encoder, dummy_inputs, dynamic_axes, output_names, output)

    if data_file_output:
        onnx_model = onnx.load(output)
        convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=data_file_output,
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save_model(onnx_model, output)

    if onnxruntime_exists:
        ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
        ort_session = onnxruntime.InferenceSession(output)  # type: ignore
        _ = ort_session.run(None, ort_inputs)
        print("Encoder has successfully been run with ONNXRuntime.")

    return output


def export_decoder(sam: Sam, output: str) -> str:
    """Export the decoder model to ONNX file.

    Args:
        sam: The SAM model.
        output: The path to save the exported ONNX file.
    Returns:
        The path to the exported ONNX file.
    """
    onnx_model = SamOnnxModel(model=sam, return_single_mask=RETURN_SINGLE_MASK)

    dynamic_axes = {
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    export_to_onnx(onnx_model, dummy_inputs, dynamic_axes, output_names, output)

    if onnxruntime_exists:
        ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)  # type: ignore
        _ = ort_session.run(None, ort_inputs)
        print("Decoder has successfully been run with ONNXRuntime.")

    return output


def add_to_cluster(exported_paths: Tuple[str, str], cluster_type: str = "local"):
    """Add the exported ONNX models to the FarmVibes.AI cluster.

    Args:
        exported_paths: A tuple of paths to the exported ONNX models.
        cluster_type: The type of cluster to add the models to (local or remote).
    """

    if cluster_type not in ["local", "remote"]:
        raise ValueError(f"Invalid cluster type: {cluster_type}. Expected 'local' or 'remote'.")

    for path in exported_paths:
        print(f"Adding {path} to cluster...")
        subprocess.run(
            [
                "farmvibes-ai",
                cluster_type,
                "add-onnx",
                path,
            ],
            check=True,
        )


def main():
    args = parse_args()

    with TemporaryDirectory() as tmp_dir:
        for model_type in args.models:
            model_url = MODELS[model_type].url
            downloaded_path = download_file(model_url, os.path.join(tmp_dir, f"{model_type}.pth"))
            exported_paths = export_model(model_type, downloaded_path, tmp_dir)
            add_to_cluster(exported_paths, args.cluster)


if __name__ == "__main__":
    main()
