import dotenv
from src.data import convert_maestro_to_tokens, convert_snes_to_tokens
import sys
import os

"""
This scripts downloads the MAESTRO dataset and converts it to the proto format.
"""

current_folder = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_folder, "data")
snes_dir = os.path.join(current_folder, "midi/snes")

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "snes":
            convert_snes_to_tokens(snes_dir, data_dir)
        elif sys.argv[1] == "maestro":
            convert_maestro_to_tokens(data_dir)
        else:
            raise ValueError("Unknown dataset!")
