from base import BaseDataLoader
from data_loader.dataset import VideoFrameAndMaskDataset, CelebAFrameAndMaskDataset
from data_loader.dataset import Places2FrameAndMaskDataset, VideoSuperResolutionDataset
from utils.directory_IO import (
    RootInputDirectories, RootOutputDirectories
)


class MaskedFrameDataLoader(BaseDataLoader):
    def __init__(
        self, root_videos_dir, root_masks_dir, root_outputs_dir,
        dataset_args,
        batch_size, shuffle, validation_split,
        num_workers, video_names_filename=None, training=True, name=None
    ):
        # Input directories
        self.rids = RootInputDirectories(
            root_videos_dir, root_masks_dir, video_names_filename=video_names_filename)

        # Output directories
        self.rods = RootOutputDirectories(root_outputs_dir)
        self.name = name

        # Dataset, default is video
        if 'type' not in dataset_args:
            dataset_args['type'] = 'video'
        if dataset_args['type'] == 'video':
            Dataset = VideoFrameAndMaskDataset
        elif dataset_args['type'] == 'CelebA':
            Dataset = CelebAFrameAndMaskDataset
        elif dataset_args['type'] == 'Places2':
            Dataset = Places2FrameAndMaskDataset
        elif dataset_args['type'] == 'super_resolution':
            Dataset = VideoSuperResolutionDataset
        else:
            raise NotImplementedError(f"Dataset type {dataset_args['type']}")
        self.dataset = Dataset(
            self.rids, self.rods, dataset_args,
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
