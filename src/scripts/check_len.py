import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA

from utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-rd', '--root_dir',
        type=str,
        required=True,
        help="Root background frames directory"
    )
    parser.add_argument(
        '-dl', '--dirname_list',
        type=str,
        required=True,
        help="Root background frames directory"
    )
    parser.add_argument(
        '-l', '--length',
        type=int
    )
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"Checking {args.root_dir}")
    with open(args.dirname_list, 'r') as fin:
        dirnames = [line.split()[0] for line in fin.readlines()]
    # dirnames = os.listdir(args.root_dir)
    for i, dirname in enumerate(dirnames):
        if i % 2000 == 0:
            logger.info(f"Checking no. {i}")
        dirpath = os.path.join(args.root_dir, dirname)
        filenames = os.listdir(dirpath)
        if len(filenames) != args.length:
            logger.error(f"{dirpath} len {len(filenames)}")


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(args)
