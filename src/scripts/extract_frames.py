import threading
import signal
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA

import cv2

from utils.util import make_dirs
from utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rid', '--root_input_directory', type=str)
    parser.add_argument('-od', '--output_directory', type=str)
    parser.add_argument('-fn', '--filename', type=str)
    parser.add_argument('-n', '--thread_number', type=int)
    parser.add_argument('-max_len', '--max_len', type=int, default=120)
    parser.add_argument('-min_w', '--min_w', type=int, default=320)
    parser.add_argument('-min_h', '--min_h', type=int, default=180)
    args = parser.parse_args()
    return args


class Manager:
    def __init__(self, filepaths):
        self.lock = threading.Lock()
        self.filepaths = filepaths


class Worker(threading.Thread):
    def __init__(self, name, manager):
        super().__init__()
        self.name = str(name)
        self.manager = manager

    def run(self):
        while len(self.manager.filepaths) > 0:
            with self.manager.lock:
                filepath = self.manager.filepaths.pop()
            dirname = filepath.split('/')[-2]
            logger.info(f"Thread {self.name} processing {filepath}")
            save_video_to_frames(
                filepath, args.output_directory, args.max_len,
                args.min_h, args.min_w,
                prefix=f"class_{dirname}_"
            )


def read_filepaths(args):
    dirnames = sorted(os.listdir(args.root_input_directory))
    logger.info(f"Total {len(dirnames)} dirs: {dirnames}")
    all_filepaths = []
    for dirname in dirnames:
        dirpath = os.path.join(args.root_input_directory, dirname)
        if not os.path.isdir(dirpath):
            continue
        logger.info(f"------------------------- class {dirname}")
        filenames = sorted(os.listdir(dirpath))
        logger.info(f"------------------------- Total {len(filenames)} files")

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            all_filepaths.append(filepath)
    return all_filepaths


def main(args):
    logger.info(args)
    filepaths = read_filepaths(args)

    manager = Manager(filepaths)

    def signal_handler(signum, frame):
        manager.filepaths = []
        logger.error(
            f"Got ctrl+c, set manager filepaths = [], please wait until all workers are done"
        )

    signal.signal(signal.SIGINT, signal_handler)

    for i in range(args.thread_number):
        worker = Worker(i, manager)
        worker.start()

    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()


def save_video_to_frames(video_filename, output_dir, max_len, min_h, min_w, prefix=''):
    video_name = prefix + video_filename.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_filename)
    frame_count = 1
    video_dir = os.path.join(output_dir, video_name)
    while frame_count <= max_len:
        ret, img = cap.read()
        if not ret:
            logger.warning(f"{video_filename} len {frame_count} < max_len {max_len}")
            break
        h, w, c = img.shape
        if h < min_h or w < min_w:
            logger.warning(f"h {h} < min_h {min_h} or w {w} < min_w {min_w}")
            break
        make_dirs(video_dir)
        output_filename = os.path.join(video_dir, f"{frame_count:04d}.png")
        logger.debug(f"  Saving {output_filename}")
        cv2.imwrite(output_filename, img)
        frame_count += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
