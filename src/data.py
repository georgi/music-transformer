from copy import deepcopy
import os
from pathlib import Path
import random
import tarfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import requests
from tqdm import tqdm
import zipfile
from miditok import MIDITokenizer, REMI, Structured, TokSequence, TokenizerConfig
from miditok.utils import get_midi_programs
from tqdm import tqdm
from miditoolkit import MidiFile
from miditok.data_augmentation import augment_midi


maestro_zip = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
maestro_csv = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
lmd_clean_tgz = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"


class MidiConverter:
    """
    A class to convert MIDI files to tokenized MIDI files.

    It does following steps:
        1. Read the MIDI file.
        2. Augment the MIDI file.
        3. Tokenize the MIDI file.
        4. Save the tokenized MIDI file.
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

    def read_midi(self, midi_file: str) -> MidiFile:
        """
        Read a MIDI file.
        """
        return MidiFile(os.path.join(self.src_dir, midi_file))

    def tokenize_midi(self, midi: MidiFile) -> TokSequence:
        res = self.tokenizer.midi_to_tokens(midi)
        if isinstance(res, TokSequence):
            return res
        else:
            return res[0]

    def timestretch_midi(self, midi: MidiFile, scale: float) -> MidiFile:
        """
        Timestretch a MIDI file.

        Args:
            midi: The MIDI file to timestretch.
            scale: The timestretch scale.
        """
        midi = deepcopy(midi)
        for track in midi.instruments:
            for note in track.notes:
                note.start = int(note.start * scale)
                note.end = int(note.end * scale)
        return midi

    def augment_midi(self, midi: MidiFile) -> list[tuple[str, MidiFile]]:
        """
        Augment a MIDI file. Returns a list of tuples containing the name of the augmentation and the
        augmented MIDI file.

        Args:
            midi: The MIDI file to augment.
        """
        augmentations = augment_midi(
            midi=midi,
            tokenizer=self.tokenizer,
            pitch_offsets=[-3, -2, -1, 0, 1, 2, 3],
            velocity_offsets=[-5, 0, 5],
            all_offset_combinations=False,
        )
        time_stretched = [
            (f"time_{stretch}", self.timestretch_midi(midi, stretch))
            for stretch in [0.95, 1.05, 0.975, 1.025, 0.9, 0.8]
        ]
        files = [("_".join(map(str, aug)), midi) for aug, midi in augmentations]
        return files + time_stretched

    def save_tokens(
        self, seq: TokSequence, token_file: str, programs: list[tuple[int, bool]]
    ):
        """
        Save a tokenized MIDI file.

        Args:
            seq: The tokenized MIDI file.
            token_file: The path to save the tokenized MIDI file to.
            programs: The programs used in the MIDI file.
        """
        self.tokenizer.save_tokens(seq, token_file, programs)

    def __call__(self, row: tuple[str, str]):
        split, midi_filename = row
        midi_file = os.path.join(self.src_dir, midi_filename)
        midi_basename = os.path.splitext(os.path.basename(midi_filename))[0]
        midi = self.read_midi(midi_file)
        programs = get_midi_programs(midi)
        if split == "train":
            for aug, augmented_midi in self.augment_midi(midi):
                seq = self.tokenize_midi(augmented_midi)
                out_file = os.path.join(
                    self.data_dir, split, midi_basename + "_" + aug + ".json"
                )
                self.save_tokens(seq, out_file, programs)
        else:
            seq = self.tokenize_midi(midi)
            out_file = os.path.join(self.data_dir, split, midi_basename + ".json")
            self.save_tokens(seq, out_file, programs)


def convert_snes_to_tokens(src_dir: str, data_dir: str, max_workers: int = 10):
    """
    Convert the SNES dataset to tokenized MIDI files.

    Args:
        data_dir: The directory to save the dataset to.
        max_workers: The number of workers to use for the conversion.
    """
    tokenizer = REMI()

    midi_converter = MidiConverter(tokenizer, src_dir, data_dir)

    rows = []
    for split in ["train", "test", "validation"]:
        split_dir = os.path.join(src_dir, split)
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
        for midi_file in os.listdir(split_dir):
            rows.append((split, os.path.join(split, midi_file)))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(midi_converter, rows), total=len(rows)):
            pass


def convert_maestro_to_tokens(data_dir: str, max_workers: int = 10):
    """
    Convert the MAESTRO dataset to tokenized MIDI files.

    Args:
        data_dir: The directory to save the dataset to.
        max_workers: The number of workers to use for the conversion.
    """
    config = TokenizerConfig(beat_res={(0, 4): 32, (4, 12): 8})
    tokenizer = Structured(config)

    src_dir = os.path.join(data_dir, "maestro-v3.0.0")
    csv_file = os.path.join(src_dir, "maestro-v3.0.0.csv")
    zip_file = os.path.join(data_dir, "maestro-v3.0.0.zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(src_dir):
        download_file(maestro_zip, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    df: pd.DataFrame = pd.read_csv(csv_file)
    for split in ["train", "test", "validation"]:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)

    midi_converter = MidiConverter(tokenizer, src_dir, data_dir)  # type: ignore

    rows = [(row.split, row.midi_filename) for row in df.itertuples()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(midi_converter, rows), total=len(rows)):
            pass


def download_file(url: str, fname: Path):
    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc="Downloading " + fname.name,
    )

    with open(fname, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()


def download_maestro(dest_dir: Path):
    """
    Download and extract the MAESTRO dataset.

    Args:
        dest_dir: The directory to save the dataset to.
    """
    if (
        dest_dir.exists()
        and dest_dir.is_dir()
        and dest_dir.joinpath("maestro-v3.0.0").exists()
    ):
        return
    os.makedirs(dest_dir, exist_ok=True)
    download_file(maestro_zip, dest_dir / "maestro.zip")
    with zipfile.ZipFile(dest_dir / "maestro.zip", "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def download_lmd_clean(dest_dir: Path):
    """
    Download and extract the Lakh MIDI Dataset (clean MIDI files).

    Args:
        dest_dir: The directory to save the dataset to.
    """
    if dest_dir.exists() and dest_dir.is_dir():
        return
    os.makedirs(dest_dir, exist_ok=True)
    download_file(lmd_clean_tgz, dest_dir / "lmd_clean.tgz")

    with tarfile.open(dest_dir / "lmd_clean.tgz", "r:gz") as tar:
        tar.extractall(dest_dir, filter=lambda x, _: x if x.isreg() else None)
