import random
import torch
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from note_seq.protobuf.music_pb2 import NoteSequence


def load_sequence(fname):
    with open(fname, 'rb') as f:
        ns = NoteSequence()
        ns.ParseFromString(f.read())  # type: ignore
        return ns


def find_files_by_extensions(root, exts=[]):
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
    if len(seq) <= max_seq:
        x = torch.full((max_seq, ), token_pad, dtype=torch.long)
        x[:len(seq)] = seq
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


def save_sequence(ns, path):
    with open(path, 'wb') as f:
        f.write(ns.SerializeToString())


def convert_midi_to_proto(midi_encoder, src, dest_dir):
    res = []
    for i, ns in enumerate(midi_encoder.load_midi(src)):
        fname = os.path.join(dest_dir, os.path.basename(src) + f'-{i}.pb')
        save_sequence(ns, fname)
        res.append(fname)
    return res


def convert_midi_to_proto_folder(midi_encoder, src_dir, dest_dir, max_workers=10):
    files = list(find_files_by_extensions(src_dir, ['.mid', '.midi']))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        res = []
        futures = [
            executor.submit(
                convert_midi_to_proto, midi_encoder, f, dest_dir)
            for f in files
        ]
        for future in tqdm(futures):
            res.extend(future.result())

