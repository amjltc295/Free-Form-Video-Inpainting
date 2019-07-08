import random

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from PIL import Image, ImageFilter
import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray

from utils.readers import (
    FrameReader, SegmentationReader
)
from utils.directory_IO import (
    RootOutputDirectories, RootInputDirectories, VideoDirectories
)
from utils.util import read_filenames_from_dir, get_everything_under
from data_loader.transform import (
    GroupScale, GroupRandomCrop, Stack, ToTorchFormatTensor,
    GroupRandomHorizontalFlip
)
from utils.logging_config import logger


class VideoFrameAndMaskDataset(Dataset):

    def __init__(
        self,
        rids: RootInputDirectories,
        rods: RootOutputDirectories,
        args: dict,
    ):
        self.rids = rids
        self.video_dirnames = rids.video_dirnames
        self.mask_dirnames = rids.mask_dirnames
        self.rods = rods
        self.sample_length = args['sample_length']
        self.random_sample = args['random_sample']
        self.random_sample_mask = args['random_sample_mask']
        # self.sample_period = args['sample_period'] if 'sample_period' in args else 1
        self.random_sample_period_max = args.get('random_sample_period_max', 1)

        self.guidance = args.get('guidance', "none")
        self.sigma = args.get('edge_sigma', 2)

        self.size = self.w, self.h = (args['w'], args['h'])
        self.mask_type = args['mask_type']

        self.do_augment = args.get('do_augment', False)
        self.skip_last = args.get('skip_last', False)

        self.mask_dilation = args.get('mask_dilation', 0)

        self._transform = transforms.Compose([
            GroupScale((int(self.h * 1.2), int(self.w * 1.2))),
            GroupRandomCrop((self.h, self.w)),
            GroupRandomHorizontalFlip(),
        ])
        self._to_tensors = transforms.Compose([
            GroupScale((self.h, self.w)),
            Stack(),
            ToTorchFormatTensor(),
        ])

        self.data_len = len(self.rids)
        if self.skip_last:
            self.data_len -= 1

    def __len__(self):
        return self.data_len

    def _get_sample_index_from_video(self, length):
        if self.random_sample:
            max_start = max(0, length - self.sample_length - 1)
            start = random.randint(0, max_start)
        else:
            start = 0
        end = start + self.sample_length
        return start, end

    def _get_masks(self, size, start_idx, end_idx, fg_dir):
        input_len = end_idx - start_idx
        if self.mask_type == 'fg':
            masks = SegmentationReader(fg_dir)[start_idx:end_idx]
            if self.mask_dilation > 0:
                masks = [m.filter(ImageFilter.MinFilter(self.mask_dilation)) for m in masks]
        elif self.mask_type == 'as_video':
            masks = FrameReader(fg_dir)[start_idx: end_idx]
        elif self.mask_type == 'random':
            masks = FrameReader(fg_dir)
            # This is to sample a random clip of the mask video.
            # If gt video is longer than mask video, mask_type="random" may yeild zero masks, then
            # using mask_type='random_segment can solve it'
            max_start = max(0, len(masks) - input_len - 1)
            start = random.randint(0, max_start)
            masks = masks[start: start + input_len]
        elif self.mask_type == 'from_start':
            masks = FrameReader(fg_dir)[0: end_idx - start_idx]
        else:
            raise NotImplementedError(f"Mask type {self.mask_type} not exists")
        if len(masks) != input_len:
            assert len(masks) < input_len
            # when mask video is shorter than input video, repeat last mask frame
            masks += [masks[-1]] * (input_len - len(masks))
        return masks

    def _process_vds(self, vds):
        sample_period = random.randint(1, self.random_sample_period_max)
        gt_reader = FrameReader(vds.frames_dir, sample_period=sample_period)
        # print(f"s{sample_period}, len{len(gt_reader)}")
        video_length = len(gt_reader)
        start, end = self._get_sample_index_from_video(video_length)
        gt_frames = gt_reader[start:end]
        if len(gt_frames) < self.sample_length:
            logger.warning(
                f"len frames {len(gt_frames)} reader {len(gt_reader)} < sample_length {self.sample_length}"
                f" dir {vds.frames_dir}")
        masks = self._get_masks(self.size, start, end, vds.mask_dir)

        if self.do_augment:
            gt_frames = self._transform(gt_frames)

        # Edge guidance
        guidances = []
        if self.guidance == "edge":
            for frame in gt_frames:
                edge = canny(rgb2gray(np.array(frame)), sigma=self.sigma)
                edge = Image.fromarray(edge.astype(np.uint8))
                guidances.append(edge)
            guidances = self._to_tensors(guidances)
        elif self.guidance == "landmarks":
            from utils.face import get_landmarks_contour
            for frame in gt_frames:
                edge = get_landmarks_contour(np.array(frame))
                edge = Image.fromarray(edge.astype(np.uint8))
                guidances.append(edge)
            guidances = self._to_tensors(guidances)

        # To tensors
        gt_tensors = self._to_tensors(gt_frames)
        mask_tensors = self._to_tensors(masks)[:video_length]

        # Deal with VOR test set problem: some ground truth videos are longer than masks
        if gt_tensors.shape[0] != mask_tensors.shape[0]:
            assert gt_tensors.shape[0] > mask_tensors.shape[0]
            gt_tensors = gt_tensors.narrow(0, 0, mask_tensors.shape[0])

        # Mask input
        input_tensors = gt_tensors * mask_tensors

        return {
            "input_tensors": input_tensors,
            "mask_tensors": mask_tensors,
            "gt_tensors": gt_tensors,
            "guidances": guidances
        }

    def _get_mask_name(self, index):
        if self.random_sample_mask:
            mask_name = random.choice(self.mask_dirnames)
        else:
            mask_name = self.mask_dirnames[index]
        return mask_name

    def __getitem__(self, index):
        video_name = self.video_dirnames[index]
        mask_name = self._get_mask_name(index)

        vds = VideoDirectories(
            self.rids, self.rods, video_name, mask_name
        )
        return self._process_vds(vds)


class CelebAFrameAndMaskDataset(VideoFrameAndMaskDataset):
    def __init__(
        self,
        rids: RootInputDirectories,
        rods: RootOutputDirectories,
        args: dict,
    ):
        super().__init__(rids, rods, args)
        self.image_dir = rids.root_videos_dir
        self.mask_dir = rids.root_masks_dir
        self.image_filenames = read_filenames_from_dir(self.image_dir, self.__class__.__name__)
        # The dirname here is actually fileanmes
        self.mask_dirnames = read_filenames_from_dir(self.mask_dir, self.__class__.__name__)
        self.crop_rect = (0, 0, args['w'], args['h'])

        self.data_len = len(self.image_filenames)

    def __getitem__(self, index):
        # Get image data like a video with length=1
        gt_frames = [Image.open(self.image_filenames[index]).crop(self.crop_rect)]
        masks = [Image.open(self._get_mask_name(index)).crop(self.crop_rect)]

        # To tensors
        gt_tensors = self._to_tensors(gt_frames)
        mask_tensors = self._to_tensors(masks)

        # Mask input
        input_tensors = gt_tensors * mask_tensors

        return {
            "input_tensors": input_tensors,
            "mask_tensors": mask_tensors,
            "gt_tensors": gt_tensors,
        }


class Places2FrameAndMaskDataset(VideoFrameAndMaskDataset):
    def __init__(
        self,
        rids: RootInputDirectories,
        rods: RootOutputDirectories,
        args: dict,
    ):
        super().__init__(rids, rods, args)
        self.image_dir = rids.root_videos_dir
        self.mask_dir = rids.root_masks_dir
        self.max_num = args.get('max_num', None)
        self.image_filenames = get_everything_under(self.image_dir, pattern="*/*.jpg")[:self.max_num]
        self.mask_filenames = get_everything_under(self.mask_dir, pattern="*/*.png")[:self.max_num]
        if len(self.image_filenames) > len(self.mask_filenames):
            logger.warning(f"image num {len(self.image_filenames)} > mask num {len(self.mask_filenames)}")

        self.data_len = len(self.image_filenames)
        self.mask_len = len(self.mask_filenames)

    def __getitem__(self, index):
        # Places2 are 256 * 256, and here we resize it to w * h
        gt_frames = [Image.open(self.image_filenames[index]).convert("RGB").resize(self.size)]  # Some are L mode
        mask_filename = random.choice(self.mask_filenames) \
            if self.random_sample_mask else self.mask_filenames[index % self.mask_len]
        masks = [Image.open(mask_filename).resize(self.size)]

        # Get image data like a video with length=1
        # To tensors
        gt_tensors = self._to_tensors(gt_frames)
        mask_tensors = self._to_tensors(masks)

        # Mask input
        input_tensors = gt_tensors * mask_tensors

        return {
            "input_tensors": input_tensors,
            "mask_tensors": mask_tensors,
            "gt_tensors": gt_tensors,
        }


class VideoSuperResolutionDataset(VideoFrameAndMaskDataset):
    def __init__(
        self,
        rids: RootInputDirectories,
        rods: RootOutputDirectories,
        args: dict
    ):
        self.upsample_rates = args.pop('upsample_rates')
        assert [rate >= 1.0 for rate in self.upsample_rates]
        super().__init__(rids, rods, args)
        self.black_mask = Image.new(mode='1', size=(self.size[0], self.size[1]), color=0)
        self.spatial_sr_mask = self._get_spatial_sr_mask()

    def _get_spatial_sr_mask(self):
        mask = Image.new(mode='1', size=(self.size[0], self.size[1]), color=0)
        pixels = mask.load()
        for i in np.arange(0, self.size[0], self.upsample_rates[1]):
            round_i = int(np.round(i))
            for j in np.arange(0, self.size[1], self.upsample_rates[2]):
                round_j = int(np.round(j))
                pixels[round_i, round_j] = 1
        return mask

    def _get_masks(self, size, start_idx, end_idx, fg_dir):
        revealed_ts = [
            int(round(t))
            for t in np.arange(0, self.sample_length, self.upsample_rates[0])
        ]
        masks = [
            self.spatial_sr_mask if t in revealed_ts
            else self.black_mask
            for t in range(self.sample_length)
        ]
        return masks
