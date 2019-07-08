"""
This script move all masks(with different ratio) in the directories which meet the pattern
in --srcs "<the pattern>" to the destination directory.

Reason: you may try several times for different parameters in gen_masks.py to get  the
desired distribution of masked ratio, and finally you will want to merge them into one
directory to form a 'mask base'.
"""
import os
import shutil
from glob import glob

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-srcs', type=str)
    parser.add_argument('-dst', type=str)
    args = parser.parse_args()
    assert args.srcs is not None
    assert args.dst is not None
    return args


if __name__ == '__main__':
    args = parse_args()

    # srcs = glob('./rand_curve*')
    # dst = './all_rand_curve'
    srcs = glob(args.srcs)
    dst = args.dst
    if dst in srcs:
        srcs.remove(dst)

    print(f'srcs: {srcs}')
    print(f'dst: {dst}')
    response = input('Right? [y/N] ')
    if response != 'y':
        exit()

    if not os.path.exists(dst):
        os.mkdir(dst)

    for ratio in range(5, 100, 10):
        sub_dir = os.path.join(dst, str(ratio))
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        original_size = len(glob(os.path.join(sub_dir, '*')))

        same_ratio_videos = []
        for d in srcs:
            same_ratio_videos += glob(os.path.join(d, str(ratio), '*'))
        same_ratio_videos = sorted(same_ratio_videos)
        print(f'Totally {len(same_ratio_videos)} video masks in ratio {ratio}')

        for i, d in enumerate(same_ratio_videos):
            shutil.move(d, os.path.join(sub_dir, f'{i + original_size}'))

    for src in srcs:
        shutil.rmtree(src)
