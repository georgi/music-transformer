from typing import List
from note_seq.midi_io import (
    midi_to_note_sequence,
)
import pretty_midi
from note_seq.sequences_lib import (
    apply_sustain_control_changes,
    split_note_sequence_on_silence,
    quantize_note_sequence_absolute,
    stretch_note_sequence,
    transpose_note_sequence,
)
from note_seq import (
    PerformanceOneHotEncoding,
    Performance,
)
from note_seq.protobuf.music_pb2 import NoteSequence


class PerformanceEncoder:
    """
    This Python module contains the class `PerformanceEncoder` which is designed to 
    perform the task of encoding and decoding musical performances. The class uses a 
    number of functionalities from the `note_seq` library, a Python library created 
    by the Magenta project at Google, designed for music generation tasks.

    The `PerformanceEncoder` class primarily consists of two methods: `encode_note_sequence` 
    and `decode_ids`. The `encode_note_sequence` method takes a `NoteSequence` object and
    converts it into a list of integer IDs. Conversely, the `decode_ids` method takes a 
    list of integer IDs and converts it back into a `NoteSequence`. The class also includes 
    a method `load_midi` to load and preprocess MIDI files.

    Key details about the `PerformanceEncoder` class:

    1. **Steps Per Second**: This is a parameter that determines how the note sequence is 
        quantized. It represents the number of time steps per second.

    2. **Reserved IDs**: The class reserves a few IDs for special tokens like padding, 
        start of sequence, and end of sequence.

    3. **Velocity Bins**: The number of bins used to quantize note velocities. The velocity 
        of a note represents the force or speed with which a key is pressed.

    4. **PerformanceOneHotEncoding**: This is an encoding scheme used to convert event objects into integer IDs.

    5. **Vocabulary Size**: This is the total number of unique IDs that can be generated, 
        which is the sum of the number of classes in the `PerformanceOneHotEncoding` and the number of reserved IDs.

    In the `load_midi` method, multiple versions of each musical sequence are created by 
    transposing and stretching the original sequence. This could be used as a data augmentation 
    technique to generate a more robust model.

    The `PerformanceEncoder` class encapsulates the conversion between musical performances and 
    a format that can be processed by machine learning algorithms. It provides the necessary tools 
    for transforming musical data into a form suitable for machine learning tasks, such as music 
    generation or music understanding.
    """
    steps_per_second: int
    num_reserved_ids: int
    token_pad: int
    token_sos: int
    token_eos: int
    num_velocity_bins: int
    encodind: PerformanceOneHotEncoding
    vocab_size: int

    def __init__(
        self,
        num_velocity_bins: int,
        steps_per_second: int,
    ):
        super(PerformanceEncoder, self).__init__()
        self.steps_per_second = steps_per_second
        self.num_reserved_ids = 4
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2
        self.num_velocity_bins = num_velocity_bins
        self.encoding = PerformanceOneHotEncoding(num_velocity_bins)
        self.vocab_size = self.encoding.num_classes + self.num_reserved_ids

    def encode_note_sequence(self, ns: NoteSequence) -> List[int]:
        performance = Performance(
            quantize_note_sequence_absolute(
                ns,
                self.steps_per_second),
            num_velocity_bins=self.num_velocity_bins
        )

        event_ids = [self.token_sos]

        for event in performance:
            id = self.encoding.encode_event(event) + self.num_reserved_ids
            if id > 0:
                event_ids.append(id)

        assert(max(event_ids) < self.vocab_size)
        assert(min(event_ids) >= 0)

        return event_ids + [self.token_eos]

    def decode_ids(self, ids: List[int]) -> NoteSequence:
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        performance = Performance(
            steps_per_second=self.steps_per_second,
            num_velocity_bins=self.num_velocity_bins
        )

        for i in ids:
            if i >= self.num_reserved_ids:
                performance.append(
                    self.encoding.decode_event(i - self.num_reserved_ids)
                )

        return performance.to_sequence()

    def load_midi(self, path: str) -> List[NoteSequence]:
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            print("Failed to load MIDI file", path, e)
            return []
        ns = midi_to_note_sequence(midi)
        ns = apply_sustain_control_changes(ns)
        # after applying sustain, we don't need control changes anymore
        del ns.control_changes[:]
        seqs = split_note_sequence_on_silence(ns)
        res = []
        for seq in seqs:
            for pitch in [-2, -1, 0, 1, 2]:
                for stretch in [0.95, 0.975, 1.0, 1.025, 1.05]:
                    ns = transpose_note_sequence(seq, pitch)[0]
                    ns = stretch_note_sequence(ns, stretch)
                    res.append(ns)
        return res
