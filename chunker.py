from pathlib import Path
from miditok import MIDITokenizer
from miditok.pytorch_data import split_midis_for_training

current_folder = Path(__file__).parent
data_dir = current_folder / "data"
chunk_dir = current_folder / "data" / "chunks"

tokenizer = MIDITokenizer.from_pretrained(
    "mgeorgi/music-transformer",
    private=True,
)

for split in ["train-augmented", "test", "validation"]:
    midi_paths = list((data_dir / split).glob("**/*.midi")) + list(
        (data_dir / split).glob("**/*.mid")
    )
    dataset_chunks_dir = chunk_dir / split.replace("-augmented", "")

    split_midis_for_training(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        save_dir=dataset_chunks_dir,
        max_seq_len=2048, # TODO: get this from config
    )
