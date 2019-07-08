"""
The original image files naming of face forensics is like "xxx_1.png", "xxx_2.png", ... "xxx_10.png", ... "xxx_100.png"
However, after sorted() the ordering will be "xxx_1.png", "xxx_100.png", "xxx_101.png",
... "xxx_109.png", "xxx_10.png", "xxx_100.png",

This script is to rename those image files as "xxx_00001.png", "xxx_00002.png", which
allows us to load data in the right ordering.
Usage: python reorder --target_dir <target dir>
"""
import os
import shutil
from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_dir', type=str
    )
    args = parser.parse_args()
    return args


def get_every(d):
    return sorted(glob(os.path.join(d, '*')))


if __name__ == '__main__':
    args = parse_args()
    for video in get_every(args.target_dir):
        for image in get_every(video):
            dirname = os.path.dirname(image)
            basename = os.path.basename(image)
            num = int(basename.split('_')[-1].split('.')[0])
            new_name = '_'.join(basename.split('_')[:-1]) + f'_{num:05d}.png'
            new_file = os.path.join(dirname, new_name)
            print(f'Moving {image} -> {new_file}')
            shutil.move(image, new_file)
