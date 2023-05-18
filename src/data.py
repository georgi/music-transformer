from dataclasses import dataclass
import random
import torch
import os
import pandas as pd
from typing import List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from note_seq.protobuf.music_pb2 import NoteSequence
import requests
from tqdm import tqdm
import zipfile

from .models.performance_encoder import PerformanceEncoder

maestro_zip = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
maestro_csv = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"

def load_sequence(fname: str):
    """
    Load a NoteSequence from a file.
    """
    with open(fname, "rb") as f:
        ns = NoteSequence()
        ns.ParseFromString(f.read())  # type: ignore
        return ns


def find_files_by_extensions(root: str, exts=[]):
    """
    Find all files with the given extensions in a directory.

    Args:
        root: The directory to search.
        exts: The extensions to search for.
    """

    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False

    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def trim_sequence(seq, max_seq, token_pad, random_offset=False):
    """
    Trim a sequence to a maximum length. Pad with the given token if
    the sequence is shorter.
    Truncation happens at a random offset if random_offset is True.

    Args:
        seq: The sequence to trim.
        max_seq: The maximum length of the sequence.
        token_pad: The token to pad with.
        random_offset: Whether to truncate at a random offset.
    """
    if len(seq) <= max_seq:
        x = torch.full((max_seq,), token_pad, dtype=torch.long)
        x[: len(seq)] = seq
    else:
        try:
            if random_offset:
                start = random.randint(0, len(seq) - max_seq - 1)
            else:
                start = 0
        except ValueError:
            start = 0
        end = start + max_seq + 1
        data = seq[start:end]
        x = data[:max_seq]
    return x


def save_sequence(ns: NoteSequence, path: str):
    """
    Save a NoteSequence to a file.
    """
    with open(path, "wb") as f:
        f.write(ns.SerializeToString())  # type: ignore


def convert_midi_to_proto(
    midi_encoder: PerformanceEncoder, src: str, dest_dir: str
) -> List[NoteSequence]:
    """
    Convert a MIDI file to a list of NoteSequences and save them to a directory.
    """
    res = []
    for i, ns in enumerate(midi_encoder.load_midi_sequences(src)):
        fname = os.path.join(dest_dir, os.path.basename(src) + f"-{i}.pb")
        save_sequence(ns, fname)
        res.append(fname)
    return res


def convert_midi_to_proto_folder(midi_encoder, src_dir, dest_dir, max_workers=10):
    """
    Convert a directory of MIDI files to the proto format.

    Args:
        midi_encoder: The encoder to use.
        src_dir: The directory containing the MIDI files.
        dest_dir: The directory to save the proto files.
        max_workers: The number of workers to use for the conversion.
    """
    files = list(find_files_by_extensions(src_dir, [".mid", ".midi"]))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        res = []
        futures = [
            executor.submit(convert_midi_to_proto, midi_encoder, f, dest_dir)
            for f in files
        ]
        for future in tqdm(futures):
            res.extend(future.result())


def convert_maestro_to_proto(data_dir: str, 
                             midi_encoder: PerformanceEncoder,
                             max_workers: int = 10):
    """
    Convert the MAESTRO dataset to the proto format.

    Args:
        src_dir: The directory containing the MAESTRO dataset.
        dest_dir: The directory to save the proto files.
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
        
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        res = []
        futures = [
            executor.submit(
                convert_midi_to_proto,
                midi_encoder,
                os.path.join(src_dir, row.midi_filename),  # type: ignore
                os.path.join(data_dir, row.split),  # type: ignore
            )  # type: ignore
            for row in df.itertuples()
        ]
        for future in tqdm(futures):
            res.extend(future.result())



def download_file(url: str, fname: str):
    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(
        total=total_size_in_bytes, 
        unit='iB', 
        unit_scale=True, 
        desc='Downloading ' + fname
    )

    with open(fname, 'wb') as file:
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

