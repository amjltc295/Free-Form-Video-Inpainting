"""
Apply all ground truth videos under args.input_dir with video masks under args.mask_dir, and
save them under args.output_dir
"""
from PIL import Image
import os
import numpy as np
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-id', '--input_dir',
        type=str,
    )
    parser.add_argument(
        '-md', '--mask_dir',
        type=str,
    )
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
    )
    args = parser.parse_args()
    return args


def get_every(d):
    return sorted(glob(os.path.join(d, '*')))


def apply_video(video_dir, mask_dir, out_dir):
    mask_paths = get_every(mask_dir)
    for i, image_file in enumerate(get_every(video_dir)):
        image = Image.open(image_file)
        mask = Image.open(mask_paths[i])
        masked = np.array(image) * np.expand_dims(np.array(mask), 2)

        name = os.path.basename(image_file)
        out_file = os.path.join(out_dir, name)
        Image.fromarray(masked).save(out_file)


if __name__ == '__main__':
    args = parse_args()
    for video_dir, mask_dir in zip(
        get_every(args.input_dir),
        get_every(args.mask_dir)
    ):
        video_name = os.path.basename(video_dir)
        out_dir = os.path.join(args.output_dir, video_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        apply_video(video_dir, mask_dir, out_dir)
