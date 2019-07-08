"""
This script is used to copy the first 32 frames of every video in the faceforensics dataset to a new directory.
"""
import os
import shutil
from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src', type=str
    )
    parser.add_argument(
        '--dst', type=str
    )
    parser.add_argument(
        '--num', type=int,
        default=32
    )
    args = parser.parse_args()
    return args


def get_every(d):
    return sorted(glob(os.path.join(d, '*')))


if __name__ == '__main__':
    args = parse_args()
    assert args.src != args.dst
    for video in get_every(args.src):
        video_name = os.path.basename(video)
        dst_dir = os.path.join(args.dst, video_name)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for image in get_every(video)[:args.num]:
            basename = os.path.basename(image)
            new_file = os.path.join(dst_dir, basename)
            shutil.copy(image, new_file)
