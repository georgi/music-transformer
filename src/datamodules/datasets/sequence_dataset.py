import torch
import torch.utils.data
import math
import random
from typing import List
from torch.utils.data.dataset import IterableDataset
from note_seq.protobuf.music_pb2 import NoteSequence
from ...models.performance_encoder import PerformanceEncoder


class SequenceDataset(IterableDataset):
    seq_length: int
    midi_encoder: PerformanceEncoder
    sequence_files: List[str]
    max_iter: int

    def __init__(
        self,
        sequence_files: List[str],
        seq_length: int,
        midi_encoder: PerformanceEncoder,
        max_iter: int = -1
    ):
        self.seq_length = seq_length
        self.midi_encoder = midi_encoder
        self.sequence_files = sequence_files
        self.max_iter = max_iter

    def load_seq(self, file_path: str) -> List[int]:
        with open(file_path, 'rb') as f:
            ns = NoteSequence()
            ns.ParseFromString(f.read())  # type: ignore
            return self.midi_encoder.encode_note_sequence(ns)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = 0
             iter_end = len(self.sequence_files)
        else:  # in a worker process
             per_worker = int(math.ceil(len(self.sequence_files) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = worker_id * per_worker
             iter_end = min(iter_start + per_worker, len(self.sequence_files))
        files = self.sequence_files[iter_start:iter_end]
        random.shuffle(files)
        iter_count = 0
        for seq_file in files:
            data = self.load_seq(seq_file)
            if len(data) >= self.seq_length:
                for i in range(0, len(data) - self.seq_length, self.seq_length):
                    yield torch.tensor(data[i:i + self.seq_length])
                    iter_count += 1
                    if self.max_iter != -1 and iter_count >= self.max_iter:
                        return
