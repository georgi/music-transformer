import os
import dotenv
from pathlib import Path
from miditok import REMI, MIDITokenizer, TokenizerConfig, Structured, TokSequence
from miditok.pytorch_data import DatasetMIDI, DataCollator, split_midis_for_training
from torch.utils.data import DataLoader

current_folder = Path(__file__).parent
data_dir = current_folder / "data"


config = TokenizerConfig(num_velocities=16, beat_res={(0, 4): 32, (4, 12): 8})
# tokenizer: MIDITokenizer = REMI(config)  # type: ignore
tokenizer = Structured(config)

# Train the tokenizer with Byte Pair Encoding (BPE)
midi_paths = list((data_dir / "train").glob("**/*.midi"))

tokenizer.learn_bpe(vocab_size=50000, files_paths=midi_paths)  # type: ignore
tokenizer.save_params(data_dir / "tokenizer.json")

tokenizer.push_to_hub(
    "mgeorgi/music-transformer",
    private=True,
)
