import os

from utils.util import make_dir_under_root, read_dirnames_under_root


OUTPUT_ROOT_DIR_NAMES = [
    'masked_frames',
    'result_frames',
    'optical_flows'
]


class RootInputDirectories:

    def __init__(
        self,
        root_videos_dir,
        root_masks_dir,
        video_names_filename=None
    ):
        self.root_videos_dir = root_videos_dir
        self.root_masks_dir = root_masks_dir
        if video_names_filename is not None:
            with open(video_names_filename, 'r') as fin:
                self.video_dirnames = [
                    os.path.join(root_videos_dir, line.split()[0])
                    for line in fin.readlines()
                ]

        else:
            self.video_dirnames = read_dirnames_under_root(root_videos_dir)
        self.mask_dirnames = read_dirnames_under_root(root_masks_dir)

    def __len__(self):
        return len(self.video_dirnames)


class RootOutputDirectories:

    def __init__(
        self, root_outputs_dir,
    ):
        self.output_root_dirs = {}
        for name in OUTPUT_ROOT_DIR_NAMES:
            self.output_root_dirs[name] = \
                make_dir_under_root(root_outputs_dir, name)

    def __getattr__(self, attr):
        if attr in self.output_root_dirs:
            return self.output_root_dirs[attr]
        else:
            raise KeyError(
                f"{attr} not in root_dir_names {self.output_root_dirs}")


class VideoDirectories:

    def __init__(
        self, root_inputs_dirs, root_outputs_dirs, video_name, mask_name
    ):
        self.name = f"video_{video_name}_mask_{mask_name}"
        rid = root_inputs_dirs
        rod = root_outputs_dirs

        self.frames_dir = os.path.join(rid.root_videos_dir, video_name)
        self.mask_dir = os.path.join(rid.root_masks_dir, mask_name)
        self.masked_frames_dir = os.path.join(rod.masked_frames, self.name)
        self.results_dir = os.path.join(rod.result_frames, self.name)
        self.flows_dir = os.path.join(rod.optical_flows, video_name)
