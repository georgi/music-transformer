import torch
import torch.utils.data
import math
import random
from typing import List
from torch.utils.data.dataset import IterableDataset
from note_seq.protobuf.music_pb2 import NoteSequence
from ...models.performance_encoder import PerformanceEncoder


class SequenceDataset(IterableDataset):
    """
    This Python module defines a class `SequenceDataset` which inherits from PyTorch's 
    `IterableDataset`. The purpose of the `SequenceDataset` class is to load, process, 
    and iterate over a list of MIDI sequence files for use in machine learning tasks, 
    such as training a model to generate music. 

    The class has several key attributes:

    - `seq_length`: The length of the sequences to be yielded by the iterator.
    - `midi_encoder`: An instance of `PerformanceEncoder` used to encode MIDI sequences.
    - `sequence_files`: A list of file paths of the MIDI sequence files.
    - `max_iter`: The maximum number of iterations the iterator should make.

    The class has several key methods:

    - `__init__`: Initializes the `SequenceDataset` instance with the necessary parameters, 
                  like the list of sequence files, the sequence length, the MIDI encoder, and optionally 
                  the maximum number of iterations.
    - `load_seq`: Loads and encodes a MIDI sequence from a given file path.
    - `__iter__`: Returns an iterator that yields encoded MIDI sequences of a specific length from the 
                  list of files. If the data loading happens in a worker process, the list of sequence files 
                  is evenly distributed among workers. The order of the sequence files is randomized to 
                  ensure that the model is exposed to a variety of training samples. If `max_iter` is 
                  specified and the number of yielded sequences reaches this limit, the iterator stops.

    This class is designed to be used in conjunction with PyTorch's DataLoader, which allows for efficient 
    loading of data in parallel using multi-threading.
    """
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
