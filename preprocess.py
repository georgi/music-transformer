import shutil
import dotenv
import pandas as pd
from src.data import (
    convert_snes_to_tokens,
    download_maestro,
)
from miditok.data_augmentation import augment_midi_dataset
import sys
import os
from pathlib import Path

"""
This scripts downloads the MAESTRO dataset and converts it to the proto format.
"""

current_folder = Path(__file__).parent
data_dir = current_folder / "data"
maestro_dir = data_dir / "maestro-v3.0.0"
snes_dir = current_folder / "midi" / "snes"

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Please provide a dataset to download!")

    if sys.argv[1] == "maestro":
        download_maestro(data_dir)
        df: pd.DataFrame = pd.read_csv(
            data_dir / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
        )
        for split in ["train", "test", "validation"]:
            os.makedirs(data_dir / split, exist_ok=True)

        for row in df.itertuples():
            filename = str(row.midi_filename)
            split = str(row.split)
            dirname = os.path.dirname(filename)
            os.makedirs(data_dir / split / dirname, exist_ok=True)
            shutil.copy(
                data_dir / "maestro-v3.0.0" / filename,
                data_dir / split / filename,
            )

        augment_midi_dataset(
            data_path=data_dir / "train",
            out_path=data_dir / "train-augmented",
            duration_in_ticks=True,
            pitch_offsets=[-2, -1, 0, 1, 2, 3],
            velocity_offsets=[-10, -5, 0, 5, 10],
            duration_offsets=[0, 1, 2],
        )
    else:
        raise ValueError("Unknown dataset!")
