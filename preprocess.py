from src.data import convert_maestro_to_proto, download_and_exctract_maestro

"""
This scripts downloads the MAESTRO dataset and converts it to the proto format.
"""

data_dir = "data"

if __name__ == "__main__":
    download_and_exctract_maestro(data_dir)

    convert_maestro_to_proto(data_dir)

