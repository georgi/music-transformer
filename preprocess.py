import sys

from src.data import convert_maestro_to_proto

if __name__ == '__main__':
    dataset = sys.argv[1]
    src_dir = sys.argv[2]
    dest_dir = sys.argv[3]

    convert_maestro_to_proto(src_dir, dest_dir)

