from typing import Optional, List
from collections import defaultdict
from .encoding import Encoding, Event
from note_seq.chords_encoder_decoder import (
    MajorMinorChordOneHotEncoding,
    ChordEncodingError
)
from note_seq.melody_inference import (
    infer_melody_for_sequence
)
from note_seq.sequences_lib import (
    is_quantized_sequence,
    steps_per_bar_in_quantized_sequence
)
from note_seq.protobuf.music_pb2 import NoteSequence
import pretty_midi
from collections import defaultdict
import numpy as np

from note_seq.sequences_lib import (
    quantize_note_sequence,
    split_note_sequence_on_time_changes,
)
from note_seq.midi_io import (
    midi_to_note_sequence,
)
from note_seq.chord_inference import (
    infer_chords_for_sequence
)
from note_seq.chords_lib import (
    BasicChordRenderer
)
from note_seq.protobuf.music_pb2 import NoteSequence


def get_melody_instrument(ns: NoteSequence) -> Optional[int]:
    pitch_by_instr = defaultdict(list)
    steps = defaultdict(lambda: defaultdict(lambda: 0))
    for note in ns.notes:  # type: ignore
        if not note.is_drum:
            steps[note.instrument][note.quantized_start_step] += 1
            pitch_by_instr[note.instrument].append(note.pitch)
#     mel_instruments = []
#     for inst, sim_notes in steps.items():
#         print(inst, np.mean(list(sim_notes.values())))
#         if np.mean(list(sim_notes.values())) < 2.5:
#             mel_instruments.append(inst)  
    mean_pitch = [
        (np.median(v), i)
        for i, v in pitch_by_instr.items()
#         if i in mel_instruments
    ]
    sorted_inst = [v for _, v in sorted(mean_pitch, reverse=True)]
    if len(sorted_inst) == 0:
        return None
    return sorted_inst[0]


def sequence_for(ns: NoteSequence, instruments: List[int]):
    new_ns = NoteSequence(
        total_time=ns.total_time,                # type: ignore
        quantization_info=ns.quantization_info,  # type: ignore
        ticks_per_quarter=ns.ticks_per_quarter,  # type: ignore
        time_signatures=ns.time_signatures,      # type: ignore
        tempos=ns.tempos                         # type: ignore
    )
    for note in ns.notes:  # type: ignore
        if note.instrument in instruments:
            new_ns.notes.append(note)  # type: ignore
    return new_ns


def extract_melody_chords(ns: NoteSequence) -> Optional[NoteSequence]:
    if len(ns.notes) < 10:  # type: ignore
        return None
    if len(ns.time_signatures) == 0:  # type: ignore
        return None
    time = ns.time_signatures[0]  # type: ignore
    if time.numerator != 4 or time.denominator != 4:
        return None
    melody_instrument = get_melody_instrument(ns)
    if melody_instrument is None:
        return None
    chord_seq = quantize_note_sequence(ns, steps_per_quarter=4)
    infer_chords_for_sequence(chord_seq, chords_per_bar=1)
    melody_seq = sequence_for(ns, [melody_instrument])
    inferred_melody_instrument = infer_melody_for_sequence(melody_seq)
    melody_seq = sequence_for(melody_seq, [inferred_melody_instrument])
    chord_melody_seq = quantize_note_sequence(melody_seq, steps_per_quarter=4)
    for ann in chord_seq.text_annotations:  # type: ignore
        chord_melody_seq.text_annotations.append(ann)  # type: ignore
    return chord_melody_seq


def render_chords(seq: NoteSequence) -> NoteSequence:
    chords = BasicChordRenderer()
    chords.render(seq)
    return seq


class ChordMelodyEncoder:
    chord_encoding: MajorMinorChordOneHotEncoding
    steps_per_quarter: int
    encoding: Encoding
    num_reserved_ids: int
    vocab_size: int
    token_pad: int
    token_sos: int
    token_eos: int

    def __init__(
        self,
        steps_per_quarter: int = 4,
    ):
        self.chord_encoding = MajorMinorChordOneHotEncoding()
        self.steps_per_quarter = steps_per_quarter
        self.encoding = Encoding()
        self.num_reserved_ids = 3
        self.vocab_size = self.encoding.num_classes + self.num_reserved_ids
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2

    def encode_note_sequence_to_events(self, ns: NoteSequence) -> List[Event]:
        assert(is_quantized_sequence(ns))
        notes = list(ns.notes)  # type: ignore
        last_step = max(note.quantized_start_step for note in ns.notes)  # type: ignore
        bars = []
        chords = []
        steps_per_bar = int(steps_per_bar_in_quantized_sequence(ns))

        # Generate bar events for whole track
        for i in range(0, last_step, steps_per_bar):
            bars.append((i, Event.BAR, 0))
        # Generate chord event for whole sequence
        for annotation in ns.text_annotations:  # type: ignore
            chord = annotation.text
            try:
                value = self.chord_encoding.encode_event(chord)
                chords.append((annotation.quantized_step, Event.CHORD, value))
            except ChordEncodingError:
                pass

        onsets = [
            (note.quantized_start_step, Event.NOTE_ON, note.pitch)
            for note in notes
        ]
        offsets = [
            (note.quantized_end_step, Event.NOTE_OFF, 0)
            for note in notes
        ]
        # Sort by time step 
        note_events = sorted(bars + chords + onsets + offsets)

        current_step = 0
        events = []
        for step, event_type, event_value in note_events:
            if step > current_step:
                while step > current_step + Encoding.MAX_SHIFT:
                    events.append(Event(Event.TIME_SHIFT, Encoding.MAX_SHIFT))
                    current_step += Encoding.MAX_SHIFT
                events.append(
                    Event(Event.TIME_SHIFT, int(step - current_step)))
                current_step = step
            events.append(Event(event_type, event_value))
        return events

    def encode_note_sequence(self, ns: NoteSequence) -> List[int]:
        events = self.encode_note_sequence_to_events(ns)

        ids = [self.token_sos] + [
            self.encoding.encode_event(event) + self.num_reserved_ids
            for event in events
        ] + [self.token_eos]

        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)
        return ids

    def decode_ids(self, ids: List[int], bpm: float = 120.0, velocity: int =100, max_note_duration: int =2):
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        sequence = NoteSequence()
        sequence.quantization_info.steps_per_quarter = 4  # type: ignore
        seconds_per_step = 60.0 / (self.steps_per_quarter * bpm)
        step = 0
        current_note = None

        for idx in ids:
            if idx < self.num_reserved_ids:
                continue
            event = self.encoding.decode_event(idx - self.num_reserved_ids)
            if event.event_type == Event.NOTE_ON:
                current_note = (event.event_value, step)
            elif event.event_type == Event.NOTE_OFF:
                if current_note:
                    pitch, start_step = current_note
                    current_note = None
                    if step == start_step:
                        continue
                    note = sequence.notes.add()  # type: ignore
                    note.start_time = start_step * seconds_per_step
                    note.end_time = step * seconds_per_step
                    note.quantized_start_step = start_step
                    note.quantized_end_step = step
                    if note.end_time - note.start_time > max_note_duration:
                        note.end_time = note.start_time + max_note_duration
                    note.pitch = pitch
                    note.velocity = velocity
                    note.instrument = 0
                    note.program = 0
                    note.is_drum = False
                    if note.end_time > sequence.total_time:  # type: ignore
                        sequence.total_time = note.end_time  # type: ignore
            elif event.event_type == Event.CHORD:
                ann = sequence.text_annotations.add()  # type: ignore
                ann.annotation_type = 1
                ann.text = self.chord_encoding.decode_event(event.event_value)
                ann.quantized_step = step
                ann.time = step * seconds_per_step
            elif event.event_type == Event.TIME_SHIFT:
                step += event.event_value
        return sequence

    def load_midi(self, path: str) -> List[NoteSequence]:
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            print("Failed to load MIDI file", path, e)
            return []
        ns = midi_to_note_sequence(midi)
        del ns.control_changes[:]  # type: ignore
        seqs = split_note_sequence_on_time_changes(ns)
        seqs = map(extract_melody_chords, seqs)
        seqs = filter(lambda ns: ns is not None, seqs)
        seqs = filter(lambda ns: len(ns.notes) > 20, seqs)  # type: ignore
        seqs = filter(lambda ns: len(ns.text_annotations) > 10, seqs)  # type: ignore
        return list(seqs)  # type: ignore


