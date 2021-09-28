import os
import sys

from src.performance_encoder import PerformanceEncoder
from src.chord_melody_encoder import ChordMelodyEncoder
from src.data import convert_midi_to_proto_folder

if __name__ == '__main__':
    encoder_type = sys.argv[1]
    midi_dir = sys.argv[2]

    if encoder_type == 'performnace':
        midi_encoder = PerformanceEncoder(
            num_velocity_bins=32,
            steps_per_second=100
        )
    elif encoder_type == 'chord_melody':
        midi_encoder = ChordMelodyEncoder()
    else:
        print("unknonw encoder", encoder_type)
        exit(1)

    convert_midi_to_proto_folder(midi_encoder, midi_dir, './data')
