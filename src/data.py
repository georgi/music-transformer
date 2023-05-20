from dataclasses import dataclass
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import requests
from tqdm import tqdm
import zipfile
from miditok import MIDITokenizer
from miditok.utils import get_midi_programs
from tqdm import tqdm
from miditoolkit import MidiFile


maestro_zip = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
maestro_csv = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"


def convert_midi_to_tokens(tokenizer: MIDITokenizer, midi_file: str, token_file: str):
    """
    Convert a MIDI file to a tokenized MIDI file.

    Args:
        tokenizer: The MIDI tokenizer to use.
        midi_file: The path to the MIDI file.
        token_file: The path to save the tokenized MIDI file to.
    """
    midi = MidiFile(midi_file)
    seq = tokenizer(midi)
    tokenizer.apply_bpe(seq)
    tokenizer.save_tokens(seq, token_file, get_midi_programs(midi))


class MidiConverter:
    """
    A class to convert MIDI files to tokenized MIDI files.
    """

    def __init__(self, tokenizer: MIDITokenizer, src_dir: str, data_dir: str):
        """
        Args:
            tokenizer: The MIDI tokenizer to use.
            src_dir: The directory containing the MIDI files.
            data_dir: The directory to save the tokenized MIDI files to.
        """
        self.tokenizer = tokenizer
        self.src_dir = src_dir
        self.data_dir = data_dir

    def __call__(self, row: tuple[str, str]):
        split, midi_filename = row
        convert_midi_to_tokens(
            self.tokenizer,
            os.path.join(self.src_dir, midi_filename),
            os.path.join(self.data_dir, split, os.path.basename(midi_filename)),
        )


def convert_maestro_to_tokens(
    tokenizer: MIDITokenizer, data_dir: str, max_workers: int = 10
):
    """
    Convert the MAESTRO dataset to tokenized MIDI files.

    Args:
        tokenizer: The MIDI tokenizer to use.
        data_dir: The directory to save the dataset to.
        max_workers: The number of workers to use for the conversion.
    """
    src_dir = os.path.join(data_dir, "maestro-v3.0.0")
    csv_file = os.path.join(data_dir, "maestro-v3.0.0.csv")
    zip_file = os.path.join(data_dir, "maestro-v3.0.0.zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(csv_file):
        download_file(maestro_csv, csv_file)

    if not os.path.exists(src_dir):
        download_file(maestro_zip, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    df: pd.DataFrame = pd.read_csv(csv_file)
    for split in ["train", "test", "validation"]:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)

    midi_converter = MidiConverter(tokenizer, src_dir, data_dir)

    rows = [(row.split, row.midi_filename) for row in df.itertuples()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(midi_converter, rows), total=len(rows)):
            pass


def download_file(url: str, fname: str):
    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc="Downloading " + fname,
    )

    with open(fname, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()


def download_and_exctract_maestro(dest_dir: str):
    """
    Download and extract the MAESTRO dataset.

    Args:
        dest_dir: The directory to save the dataset to.
    """
    if os.path.exists(os.path.join(dest_dir, "maestro-v3.0.0")):
        return
    os.makedirs(dest_dir, exist_ok=True)
    download_file(maestro_zip, os.path.join(dest_dir, "maestro.zip"))
    with zipfile.ZipFile(os.path.join(dest_dir, "maestro.zip"), "r") as zip_ref:
        zip_ref.extractall(dest_dir)
