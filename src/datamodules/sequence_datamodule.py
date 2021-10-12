import os
from random import randrange
from typing import Optional, List
from note_seq.protobuf.music_pb2 import NoteSequence
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .datasets.sequence_dataset import SequenceDataset
from ..models.performance_encoder import PerformanceEncoder
from ..data import load_sequence
import logging

logger = logging.getLogger(__name__)


def train_test_split(dataset, split=0.90):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def load_datasets(
    midi_encoder,
    data_dir,
    max_seq,
    time_augment,
    transpose_augment,
):
    sequences: List[NoteSequence] = [
        load_sequence(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
    ]

    logger.info(f"Found {len(sequences)} sequence files")

    sequences = [
        ns for ns in sequences
        if len(ns.notes) >= max_seq  # type: ignore
    ]

    logger.info(f"Valid sequences: {len(sequences)}")

    train_seqs, valid_seqs = train_test_split(sequences)

    train_data = SequenceDataset(
        sequences=train_seqs,
        seq_length=max_seq,
        midi_encoder=midi_encoder,
        time_augment=time_augment,
        transpose_augment=transpose_augment,
        random_offset=True
    )
    valid_data = SequenceDataset(
        sequences=valid_seqs,
        seq_length=max_seq,
        midi_encoder=midi_encoder,
        time_augment=0,
        transpose_augment=0,
        random_offset=True
    )

    return train_data, valid_data


class SequenceDataModule(LightningDataModule):
    data_dir: str
    max_seq: int
    time_augment: float
    transpose_augment: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    train_data: SequenceDataset
    val_data: SequenceDataset
    encoder: PerformanceEncoder

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        time_augment: float,
        transpose_augment: int,
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
        self.time_augment = time_augment
        self.transpose_augment = transpose_augment
        self.max_seq = max_seq
        self.encoder = PerformanceEncoder(
            num_velocity_bins=num_velocity_bins,
            steps_per_second=steps_per_second
        )


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_data, self.val_data = load_datasets(
            midi_encoder=self.encoder,
            data_dir=self.data_dir,
            max_seq=self.max_seq,
            time_augment=self.time_augment,
            transpose_augment=self.transpose_augment
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
