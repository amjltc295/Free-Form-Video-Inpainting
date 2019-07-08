import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--base_dir',
        type=str,
    )
    parser.add_argument(
        '-p', '--print_only',
        action='store_true'
    )
    parser.add_argument(
        '-pf', '--postfix',
        type=str
    )
    parser.add_argument(
        '-t', '--type',
        type=str,
        required=True,
        choices=['dir', 'file']
    )
    args = parser.parse_args()
    return args


def main(args):
    basedir = args.base_dir
    for i, name in enumerate(sorted(os.listdir(basedir))):
        path = os.path.join(basedir, name)
        if os.path.isdir(path) and args.type == 'file':
            continue
        elif os.path.isfile(path) and args.type == 'dir':
            continue
        else:
            new_path = os.path.join(
                basedir,
                f'{i:05d}{args.postfix}{"." + path.split(".")[-1] if args.type == "file" else ""}'
            )
            if not args.print_only:
                os.rename(path, new_path)
            print(f"Rename {path} ->  {new_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
