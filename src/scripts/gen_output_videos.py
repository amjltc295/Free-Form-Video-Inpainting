import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA

from utils.util import make_dirs, read_dirnames_under_root
from utils.readers import FrameReader
from utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-id', '--input_root_dir',
        type=str,
        help="Input root directory name"
    )
    parser.add_argument(
        '-od', '--output_root_dir',
        type=str,
        help="Output root directory name"
    )
    parser.add_argument(
        '-ml', '--max_len',
        type=int, default=999,
        help="Max length of the video"
    )
    parser.add_argument(
        '-ip', '--input_postfix',
        type=str,
        default='',
        help="Input dir post dirname"
    )
    args = parser.parse_args()
    return args


def main(args):
    make_dirs(args.output_root_dir)
    dirnames = read_dirnames_under_root(args.input_root_dir)
    for dirname in dirnames:
        try:
            output_path = os.path.join(args.output_root_dir, f"{dirname}.mp4")
            dirpath = os.path.join(args.input_root_dir, dirname, args.input_postfix)
            reader = FrameReader(dirpath, max_length=args.max_len)
            reader.write_files_to_video(output_path)
        except Exception as err:
            logger.error(err, exc_info=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
