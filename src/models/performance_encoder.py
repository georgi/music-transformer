from typing import List
from note_seq.midi_io import (
    midi_to_note_sequence,
)
import pretty_midi
from note_seq.sequences_lib import (
    apply_sustain_control_changes,
    split_note_sequence_on_silence,
    quantize_note_sequence_absolute,
)
from note_seq import (
    PerformanceOneHotEncoding,
    Performance,
)
from note_seq.protobuf.music_pb2 import NoteSequence


class PerformanceEncoder:
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

    def decode_ids(self, ids):
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

    def load_midi(self, path):
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            print("Failed to load MIDI file", path, e)
            return []
        ns = midi_to_note_sequence(midi)
        ns = apply_sustain_control_changes(ns)
        # after applying sustain, we don't need control changes anymore
        del ns.control_changes[:]
        return split_note_sequence_on_silence(ns)


