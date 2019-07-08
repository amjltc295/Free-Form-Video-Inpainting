"""
The script allows you to copy masks from a 'mask base' to form a dataset with
a given range of masked ratio.
"""
import os
import shutil
from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument(
        '--start', type=int,
        help=('The start index to copy for each masked ratio directory. '
              'Index range [args.start, args.start + args.num) will be used. '
              'For different mask datasets, you should avoid overlapping range.')
    )
    parser.add_argument('--num', type=int)
    parser.add_argument('--min_area', type=int, default=5)
    parser.add_argument('--max_area', type=int, default=60)
    args = parser.parse_args()
    assert args.base is not None
    assert args.dst is not None
    return args


if __name__ == '__main__':
    args = parse_args()
    # An example
    # name = 'rand_curve'
    # base = '../../../dataset/random_mask_base/all_rand_curve'
    # dst = './test_rand_curve/'
    # start = 1000
    # num = 25

    end = args.start + args.num
    print(f'name: {args.name}')
    print(f'base: {args.base}')
    print(f'dst: {args.dst}')
    print(f'args.start/end: {args.start}/{end}')

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    for ratio in range(5, 100, 10):
        if ratio > args.max_area or ratio < args.min_area:
            print(f'Area ratio {ratio} exceeds range, skipping..')
            continue

        same_ratio_videos = sorted(glob(os.path.join(args.base, str(ratio), '*')))
        print(f'Totally {len(same_ratio_videos)} video masks in ratio {ratio}')
        print(f'(Using {args.start} ~ {end})')

        for i, d in enumerate(same_ratio_videos[args.start:end]):
            shutil.copytree(d, os.path.join(args.dst, f'{args.name}_{ratio}_{i}'))
