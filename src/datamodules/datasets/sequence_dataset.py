import torch
from typing import List
from random import randrange, uniform
from torch.utils.data import Dataset
from note_seq.sequences_lib import (
    stretch_note_sequence,
    transpose_note_sequence,
    NegativeTimeError
)
from ...data import trim_sequence
from ...models.performance_encoder import PerformanceEncoder
from note_seq.protobuf.music_pb2 import NoteSequence


class SequenceDataset(Dataset):
    sequences: List[NoteSequence]
    seq_length: int
    midi_encoder: PerformanceEncoder
    time_augment: float
    transpose_augment: int
    random_offset: bool
    token_pad: int

    def __init__(
        self,
        sequences: List[NoteSequence],
        seq_length: int,
        midi_encoder: PerformanceEncoder,
        time_augment: float,
        transpose_augment: int,
        random_offset: bool,
        token_pad: int = 0
    ):
        self.sequences = sequences
        self.seq_length = seq_length
        self.midi_encoder = midi_encoder
        self.time_augment = time_augment
        self.transpose_augment = transpose_augment
        self.random_offset = random_offset
        self.token_pad = token_pad

    def __len__(self):
        return len(self.sequences)

    def augment(self, ns):
        if self.transpose_augment > 0:
            transpose = randrange(-self.transpose_augment,
                                  self.transpose_augment)
            ns = transpose_note_sequence(ns, transpose)[0]
        if self.time_augment > 0:
            try:
                stretch_factor = uniform(
                    1.0 - self.time_augment,
                    1.0 + self.time_augment
                )
                ns = stretch_note_sequence(ns, stretch_factor)
            except NegativeTimeError:
                pass
        return ns

    def encode(self, ns):
        return self.midi_encoder.encode_note_sequence(ns)

    def __getitem__(self, idx):
        return self._get_seq(self.sequences[idx])

    def _get_seq(self, ns: NoteSequence):
        data = torch.tensor(self.encode(self.augment(ns)))
        data = trim_sequence(data, self.seq_length, self.token_pad, random_offset=self.random_offset)
        return data
