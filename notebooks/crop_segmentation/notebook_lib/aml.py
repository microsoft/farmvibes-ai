import glob
import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class CropSegChipDataset(Dataset):
    """
    Dataset for AML training/inference for NDVI and CDL chips/patches stored locally.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ndvi_paths = glob.glob(os.path.join(self.data_dir, "ndvi", "*.pt"))
        self.cdl_paths = [path.replace("ndvi", "cdl") for path in self.ndvi_paths]

    def __getitem__(self, index: int):
        ndvi = torch.load(self.ndvi_paths[index])
        cdl = torch.load(self.cdl_paths[index])
        sample = {"image": ndvi, "mask": cdl}
        return sample

    def __len__(self):
        return len(self.ndvi_paths)


class CropSegChipsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """
        Init a Crop Segmentation Data Module instance for pre-generated chips

        Args:
        data_dir: dir with train and val folders where respective ndvi and cdl maps are
        stored (e.g., aml/dataset)
        batch_size: how many samples are fed to the network in a single batch.
        num_workers: how many worker processes to use in the data loader.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Skipping prepare_data as data is already downloaded
        pass

    def setup(self, stage: Optional[str] = None):
        train_dir = os.path.join(self.data_dir, "train")
        self.train_dataset = CropSegChipDataset(train_dir)

        val_dir = os.path.join(self.data_dir, "val")
        self.val_dataset = CropSegChipDataset(val_dir)

    def _get_dataloader(self, dataset: CropSegChipDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=5 if self.num_workers else 2,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
