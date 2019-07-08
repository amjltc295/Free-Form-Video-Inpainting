import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA

from utils.readers import CompareFramesReader
from utils.util import read_dirnames_under_root, make_dirs


DEFAULT_FPS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fps', '--fps',
        type=int, default=DEFAULT_FPS,
        help="Output video FPS"
    )
    parser.add_argument(
        '-vs', '--root_frames_dirs',
        nargs='+',
        type=str,
        required=True,
        help="Video directory names"
    )
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
        required=True,
        help="Output directory name"
    )
    parser.add_argument(
        '-md', '--root_mask_dir',
        type=str,
        help="Masks directory name"
    )
    parser.add_argument(
        '-rp', '--result_postfix',
        type=str,
        default='color',
        help="Result directory postfix"
    )
    parser.add_argument(
        '-names', '--names',
        nargs='+',
        type=str,
        default=['Input Video', 'Image-based Inpainting [53]', 'Patch-based Video Inpainting [33]', 'VORNet (Ours)'],
        help="Masks directory name"
    )
    parser.add_argument(
        '-n', '--test_num',
        type=int,
        default=100
    )
    parser.add_argument(
        '-fnwrl', '--frame_num_when_repeat_list',
        type=int,
        nargs='+',
        default=[1]
    )
    parser.add_argument(
        '--assume_ordered', action='store_true',
        help=("Set this if you assume the files of different frames dirs are in"
              "the same order (ignore exact video name matching)")
    )
    parser.add_argument(
        '--name_prefix', type=str, default=''
    )
    parser.add_argument(
        '--col', type=int, default=2,
        help="how many columns when showing all the videos."
    )
    args = parser.parse_args()
    return args


def main(args):
    make_dirs(args.output_dir)
    # root_frames_dirs: methods' output directories
    root_frames_dirs = args.root_frames_dirs
    if args.assume_ordered:
        # frames_dirnames[i][j]: target video directory's basename
        # i: method index, j: video index
        frames_dirnames_list = [
            read_dirnames_under_root(root_frames_dir)
            for root_frames_dir in root_frames_dirs
        ]
        for j in range(len(frames_dirnames_list[0])):
            # Find the target video directory of each methods
            targets = [
                os.path.join(root_frames_dirs[i], frames_dirnames_list[i][j])
                for i in range(len(frames_dirnames_list))
            ]
            # For each target video directory, transform its path if it contains args.result_postfix
            targets = [
                target if args.result_postfix not in os.listdir(target)
                else os.path.join(target, args.result_postfix)
                for target in targets
            ]
            reader = CompareFramesReader(targets, names=args.names, col=args.col)
            reader.write_files_to_video(
                os.path.join(args.output_dir, f"{args.name_prefix}{j:04d}_compare.mp4"),
                fps=args.fps, frame_num_when_repeat_list=args.frame_num_when_repeat_list
            )

    else:
        frames_dirnames = read_dirnames_under_root(root_frames_dirs[0])[:args.test_num]
        for name in frames_dirnames:
            reader = CompareFramesReader(
                [os.path.join(x, name)
                    if args.result_postfix not in os.listdir(os.path.join(x, name))
                    else os.path.join(x, name, args.result_postfix)
                    for x in root_frames_dirs],
                names=args.names, col=args.col
                # mask_dir=os.path.join(args.root_mask_dir, name)
            )
            reader.write_files_to_video(
                os.path.join(args.output_dir, f"{args.name_prefix}{name}_compare.mp4"),
                fps=args.fps, frame_num_when_repeat_list=args.frame_num_when_repeat_list
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
