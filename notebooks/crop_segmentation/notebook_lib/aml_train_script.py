import argparse

import mlflow
import torch
from aml import CropSegChipsDataModule
from models import ModelPlusSigmoid, SegmentationModel
from pytorch_lightning import Trainer


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--onnx_model_path", type=str)
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--ndvi_stack_bands", type=int, default=37)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args: argparse.Namespace):
    # Setup DataLoader
    data = CropSegChipsDataModule(
        data_dir=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    data.setup()

    # Setup Segmentation Model
    model = SegmentationModel(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        in_channels=args.ndvi_stack_bands,
        num_epochs=args.max_epochs,
        classes=1,
    )

    # Enables logging
    mlflow.pytorch.autolog(log_models=False)

    # Train
    trainer = Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=args.num_gpus)
    trainer.fit(model, data)

    # Signature
    batch = next(iter(data.train_dataloader()))
    ndvi_batch = batch["image"]
    ndvi_sample = ndvi_batch[0:1, :, :, :].numpy()

    # Set model to inference mode before exporting to ONNX
    trace_model = ModelPlusSigmoid(model).eval()
    dummy_input = torch.randn(
        args.batch_size, args.ndvi_stack_bands, ndvi_sample.shape[-2], ndvi_sample.shape[-1]
    )

    # Export the model
    torch.onnx.export(
        trace_model,
        dummy_input,  # model example input
        args.onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=11,
        input_names=["ndvi_stack"],  # the model's input names
        output_names=["seg_map"],  # the model's output names
        dynamic_axes={
            "ndvi_stack": {0: "batch_size"},  # variable length axes
            "seg_map": {0: "batch_size"},
        },
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
