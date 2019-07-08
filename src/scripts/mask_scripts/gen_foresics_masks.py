import os
import sys
from glob import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA

import random
import argparse
from PIL import Image, ImageDraw

from utils.util import get_everything_under
from utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-id', '--input_dir',
        type=str,
        help="Output directory name"
    )
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
        help="Output directory name"
    )
    parser.add_argument(
        '--img_size',
        type=int,
        nargs=2
    )
    parser.add_argument(
        '--n_masks',
        type=int
    )
    args = parser.parse_args()
    return args


def get_random_mask(img_size=[128, 128], r=[0.375, 0.5]):
    # Sample the square width uniformly from r[0] * img_size to r[1] * img_size
    shorter_side = min(img_size)
    ratio = random.random() * (r[1] - r[0]) + r[0]
    block_size = int(shorter_side * ratio)

    # Find a random position for mask (but not exceeding the boarder)
    x0 = random.randrange(0, img_size[0] - block_size)
    y0 = random.randrange(0, img_size[1] - block_size)
    x1, y1 = x0 + block_size, y0 + block_size

    mask = Image.new(mode='1', size=(img_size[0], img_size[1]), color=1)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=0)
    return mask


def gen_masks(in_direc, out_direc, args):
    assert in_direc != out_direc, f'input and output directory should not be the same'

    video_dirs = get_everything_under(in_direc)
    assert all([os.path.isdir(d) for d in video_dirs])

    for i, video_dir in enumerate(video_dirs):
        video_name = os.path.basename(video_dir)
        length = len(glob(os.path.join(video_dir, '*')))
        assert length > 0, f'Got video in "{video_dir}" with no frames'
        if i % 50 == 0:
            logger.info(f'generating masks of video #{i}, name: {video_name}, len: {length}')

        mask_dir = os.path.join(out_direc, video_name)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir, mode=0o755)

        mask = get_random_mask(img_size=args.img_size)
        mask_len = length if args.n_masks is None else args.n_masks
        for j in range(mask_len):
            mask_name = f'{j:05d}.png'
            mask.save(os.path.join(mask_dir, mask_name))


if __name__ == "__main__":
    args = parse_args()

    # face foresics video dataset
    # for set_ in ['test', 'train', 'val']:
    #     for type_ in ['original']:
    #         logger.info(f'Processing set {set_}, type {type_}')
    #         gen_masks(
    #             os.path.join(args.input_dir, set_, type_),
    #             os.path.join(args.output_dir, set_, type_),
    #             args
    #         )

    # VOR dataset
    gen_masks(args.input_dir, args.output_dir, args)
