import os
from typing import Optional, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .datasets.sequence_dataset import SequenceDataset
from ..models.performance_encoder import PerformanceEncoder
import logging

logger = logging.getLogger(__name__)


def load_datasets(
    midi_encoder: PerformanceEncoder,
    data_dir: str,
    max_seq: int,
):
    """
    The load_datasets function loads MIDI sequence datasets from a specified 
    directory for each data split (training, validation, and testing). 
    It uses a PerformanceEncoder to process MIDI files and caps the sequences 
    at a specified maximum length.
    """
    datasets = []
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(data_dir, split)
        sequence_files: List[str] = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
        ]

        logger.info(f"Found {len(sequence_files)} sequence files in {split_dir}")

        ds = SequenceDataset(
            sequence_files=sequence_files,
            seq_length=max_seq,
            midi_encoder=midi_encoder,
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
    used in a PyTorch Lightning training loop. It should be noted that the prepare_data 
    method is currently empty, suggesting that any preprocessing or data preparation steps
    happen outside of this class or are encapsulated within the SequenceDataset or PerformanceEncoder.
    """
    data_dir: str
    max_seq: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    train_data: SequenceDataset
    val_data: SequenceDataset
    test_data: SequenceDataset
    encoder: PerformanceEncoder

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        num_velocity_bins: int,
        steps_per_second: int,
        max_seq: int
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq = max_seq
        self.encoder = PerformanceEncoder(
            num_velocity_bins=num_velocity_bins,
            steps_per_second=steps_per_second
        )


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_data, self.val_data, self.test_data = load_datasets(
            midi_encoder=self.encoder,
            data_dir=self.data_dir,
            max_seq=self.max_seq,
        )
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
