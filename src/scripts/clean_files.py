import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA

from PIL import Image

from utils.util import read_dirnames_under_root, rm_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-rd', '--root_dir',
        type=str,
        required=True,
        help="Root background frames directory"
    )
    args = parser.parse_args()
    return args


def test_all(
    root_bg_frames_dir,
):
    bg_skip_list = [
    ]
    bg_frames_dirnames = read_dirnames_under_root(
        root_bg_frames_dir, bg_skip_list
    )

    count = 0
    for i, name in enumerate(bg_frames_dirnames):
        dirname = os.path.join(root_bg_frames_dir, name)
        for root, dirs, names in os.walk(dirname):
            if len(names) < 10:
                print(f"{name} len {len(names)}")
                count += 1
                rm_dirs(f"{root}", ignore_errors=False)
                continue
            for n in names:
                path = os.path.join(root, n)
                image = Image.open(path)
                if image.size != (1280, 720):
                    count += 1
                    print(f"{name} {image.size}")
                    rm_dirs(f"{root}", ignore_errors=False)
                break
    print(f"Removed {count} / {len(bg_frames_dirnames)}")


if __name__ == "__main__":
    args = parse_args()
    test_all(
        args.root_dir,
    )
