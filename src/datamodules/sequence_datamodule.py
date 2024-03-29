import os
from typing import Optional, List
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from .datasets.sequence_dataset import SequenceDataset
import logging

logger = logging.getLogger(__name__)


def load_datasets(
    data_dir: str,
    max_seq: int,
):
    """
    The load_datasets function loads MIDI sequence datasets from a specified
    directory for each data split (training, validation, and testing).

    Args:
        data_dir: The directory containing the tokenized MIDI sequence files.
        max_seq: The maximum sequence length.
    """
    datasets = []
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(data_dir, split)
        sequence_files: List[str] = [
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
        ]

        logger.info(f"Found {len(sequence_files)} sequence files in {split_dir}")

        ds = SequenceDataset(
            sequence_files=sequence_files,
            seq_length=max_seq,
        )
        datasets.append(ds)

    return datasets


class SequenceDataModule(LightningDataModule):
    """
    This module provides a data handling infrastructure for a machine
    learning task involving sequence data, specifically MIDI data. It is
    designed to work with the PyTorch Lightning framework.

    The SequenceDataModule class provides DataLoader instances for the training,
    validation, and testing datasets. These DataLoader instances can be directly
    used in a PyTorch Lightning training loop.
    """

    data_dir: str
    max_seq: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    train_data: SequenceDataset
    val_data: SequenceDataset
    test_data: SequenceDataset

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        max_seq: int,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq = max_seq
        self.train_data, self.val_data, self.test_data = load_datasets(
            data_dir=self.data_dir,
            max_seq=self.max_seq,
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.val_data.max_iter = 500

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
