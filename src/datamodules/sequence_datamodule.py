from pathlib import Path
from lightning.pytorch import LightningDataModule
from miditok import MIDITokenizer
from torch.utils.data import DataLoader
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok import REMI, TokenizerConfig
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SequenceDataModule(LightningDataModule):
    """
    This module provides a data handling infrastructure for a machine
    learning task involving sequence data, specifically MIDI data. It is
    designed to work with the PyTorch Lightning framework.

    The SequenceDataModule class provides DataLoader instances for the training,
    validation, and testing datasets. These DataLoader instances can be directly
    used in a PyTorch Lightning training loop.
    """

    data_dir: Path
    max_seq: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    tokenizer: MIDITokenizer

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        max_seq: int,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq = max_seq

        config = TokenizerConfig(
            use_programs=True,
            use_time_signatures=True,
            one_token_stream_for_programs=True,
            num_velocities=8,
            beat_res={(0, 4): 4, (4, 12): 4},
        )
        self.tokenizer = REMI(config)

    def get_dataloader(self, split: str) -> DataLoader:
        """
        Returns a DataLoader instance for the given dataset split.
        """
        data_dir = self.data_dir
        dataset = DatasetMIDI(
            files_paths=list(data_dir.glob("**/*.mid")),
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq,
            bos_token_id=self.tokenizer["BOS_None"],  # type: ignore
            eos_token_id=self.tokenizer["EOS_None"],  # type: ignore
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=DataCollator(self.tokenizer["PAD_None"]),  # type: ignore
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self.get_dataloader("")
