from pathlib import Path
from miditok import REMI, TokenizerConfig, MIDITokenizer
from miditok.utils import split_midi_per_tracks
from miditok.pytorch_data import split_midi_per_note_density
from symusic import Score
from tqdm import tqdm
from typing import Sequence

current_folder = Path(__file__).parent

config = TokenizerConfig(
    use_programs=True, use_time_signatures=True, one_token_stream_for_programs=True
)
tokenizer = REMI(config)


def get_average_num_tokens_per_note(
    tokenizer: MIDITokenizer, files_paths: Sequence[Path]
) -> float:
    """
    Return the average number of tokens per note (tpn) for a list of MIDIs.

    With BPE, the average tpn is likely to be very low.

    :param tokenizer: tokenizer.
    :param files_paths: list of MIDI file paths.
    :return: the average tokens per note.
    """
    num_tokens_per_note = []
    for file_path in files_paths:
        try:
            midi = Score(file_path)
            tok_seq = tokenizer(midi)
        except Exception as e:
            print(f"Error while processing {file_path}: {e}")
            continue
        if tokenizer.one_token_stream:
            num_notes = midi.note_num()
            num_tokens_per_note.append(len(tok_seq) / num_notes)
        else:
            for track, seq in zip(midi.tracks, tok_seq):
                num_tokens_per_note.append(len(seq) / track.note_num())

    return sum(num_tokens_per_note) / len(num_tokens_per_note)


def split_midis_for_training(
    files_paths: Sequence[Path],
    tokenizer: MIDITokenizer,
    save_dir: Path,
    max_seq_len: int,
    average_num_tokens_per_note: float | None = None,
    num_overlap_bars: int = 1,
    min_seq_len: int | None = None,
) -> list[Path]:
    """
    Split a list of MIDIs into smaller chunks to use for training.

    MIDI splitting allows to split each MIDI from a dataset into chunks of lengths
    calculated in function of the note densities of its bars in order to reduce the
    padding of the batches, using the
    :py:func:`miditok.pytorch_data.split_midi_per_note_density` method.
    The MIDIs are only split at bars, in order have chunks starting at relevant times.

    MIDI splitting can be performed on a dataset once. This method will save a hidden
    file, with a name corresponding to the hash of the list of file paths, in the
    ``save_dir`` directory. When called, it will first check that this file does not
    already exist, and if it is the case will return the paths to all the MIDI files
    within ``save_dir``.

    **If your tokenizer does not tokenize all tracks in one sequence of tokens**
    (``tokenizer.one_token_stream``), the MIDI tracks will be split independently.

    :param files_paths: paths to MIDI files to split.
    :param tokenizer: tokenizer.
    :param save_dir: path to the directory to save the MIDI splits.
    :param max_seq_len: maximum token sequence length that the model will be trained
        with.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer. If given ``None``, this value will automatically be calculated
        from the first 200 MIDI files with the
        :py:func:`miditok.pytorch_data.get_average_num_tokens_per_note` method.
    :param num_overlap_bars: will create chunks with consecutive overlapping bars. For
        example, if this argument is given ``1``, two consecutive MIDI chunks might
        end at the bar *n* and start at the bar *n-1* respectively, thus they will
        encompass the same bar. This allows to create a causality chain between chunks.
        This value should be determined based on the ``average_num_tokens_per_note``
        value of the tokenizer and the ``max_seq_len`` value, so that it is neither
        too high nor too low. (default: ``1``).
    :param min_seq_len: minimum sequence length, only used when splitting at the last
        bar of the MIDI. (default: ``None``, see default value of
        :py:func:`miditok.pytorch_data.split_midi_per_note_density`)
    :return: the paths to the MIDI splits.
    """
    # Safety checks
    midi_split_hidden_file_path = save_dir / f".{hash(tuple(files_paths))}"
    if midi_split_hidden_file_path.is_file():
        return list(save_dir.glob("**/*.mid"))
    if not average_num_tokens_per_note:
        average_num_tokens_per_note = get_average_num_tokens_per_note(
            tokenizer, files_paths[:200]
        )

    # Determine the deepest common subdirectory to replicate file tree
    all_parts = [path.parent.parts for path in files_paths]
    max_depth = max(len(parts) for parts in all_parts)
    root_parts = []
    for depth in range(max_depth):
        if len({parts[depth] for parts in all_parts}) > 1:
            break
        root_parts.append(all_parts[0][depth])
    root_dir = Path(*root_parts)

    # Splitting MIDIs
    new_files_paths = []
    for file_path in tqdm(
        files_paths,
        desc=f"Splitting MIDIs ({save_dir})",
        miniters=int(len(files_paths) / 20),
        maxinterval=480,
    ):

        def load_midi(file_path):
            try:
                return Score(file_path)
            except Exception as e:
                print(f"Error while loading {file_path}: {e}")
                return None

        midis = [load_midi(file_path)]
        midis = [midi for midi in midis if midi is not None]

        # Separate track first if needed
        tracks_separated = False
        if not tokenizer.one_token_stream and len(midis[0].tracks) > 1:
            midis = split_midi_per_tracks(midis[0])
            tracks_separated = True

        # Split per note density
        for ti, midi_to_split in enumerate(midis):
            try:
                midi_splits = split_midi_per_note_density(
                    midi_to_split,
                    max_seq_len,
                    average_num_tokens_per_note,
                    num_overlap_bars,
                    min_seq_len,
                )
            except Exception:
                print(f"Error while splitting {file_path}")
                continue

            # Save them
            for _i, midi_to_save in enumerate(midi_splits):
                # Skip it if there are no notes, this can happen with
                # portions of tracks with no notes but tempo/signature
                # changes happening later
                if len(midi_to_save.tracks) == 0 or midi_to_save.note_num() == 0:
                    continue
                # Add a marker to indicate chunk number
                # midi_to_save.markers.append(
                #    TextMeta(0, f"miditok: chunk {_i}/{len(midi_splits) - 1}")
                # )
                if tracks_separated:
                    file_name = f"{file_path.stem}_t{ti}_{_i}.mid"
                else:
                    file_name = f"{file_path.stem}_{_i}.mid"
                # use with_stem when dropping support for python 3.8
                saving_path = (
                    save_dir / file_path.relative_to(root_dir).parent / file_name
                )
                saving_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    midi_to_save.dump_midi(saving_path)
                    new_files_paths.append(saving_path)
                except Exception as e:
                    print(f"Error while saving {saving_path}: {e}")

    # Save file in save_dir to indicate MIDI split has been performed
    with midi_split_hidden_file_path.open("w") as f:
        f.write(f"{len(files_paths)} files after MIDI splits")

    return new_files_paths


# splits = ["train-augmented", "test", "validation"]
data_dir = current_folder / "data" / "lmd" / "clean_midi"
chunk_dir = current_folder / "data" / "lmd_chunks"


midi_paths = list(data_dir.glob("**/*.mid"))

print(f"Splitting {len(midi_paths)} MIDIs...")

split_midis_for_training(
    files_paths=midi_paths,
    tokenizer=tokenizer,
    save_dir=chunk_dir,
    max_seq_len=2048,  # TODO: get this from config
)


chunk_paths = list(chunk_dir.glob("**/*.mid"))
print(f"Validating {len(midi_paths)} chunks...")
for f in tqdm(chunk_paths):
    try:
        tokenizer.midi_to_tokens(Score(f))
    except Exception as e:
        print(f"Error while processing {f}: {e}")
        print("Removing file...")
        f.unlink()
        continue
