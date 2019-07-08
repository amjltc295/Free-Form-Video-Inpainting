## Download datasets used in our paper
Download the one you need and extract them in dataset/
- [FVI](https://drive.google.com/open?id=1leqcOfiFqu16e1T7V_68PGlOEugyuyDM) note that YouTube-Bounding-Boxes dataset is too large for google drive, so it only contains videos from YouTube-VOS.
If you want to include videos from YouTube-Bounding-Boxes dataset, please manually download it from [here](https://github.com/mbuckler/youtube-bb)
and filter out those with resolution lower than 640x480.
- [FaceForensics](https://drive.google.com/open?id=1leqcOfiFqu16e1T7V_68PGlOEugyuyDM)

**Note: the input frames and masks are matched by python sorting order.**

After the download, you should have following structures in dataset/
```
dataset
├── FaceForensics
│   ├── Test
│   │   ├── masks
│   │   └── videos
│   └── Train
│       ├── masks
│       └── videos
└── FVI
    ├── Test
    │   ├── JPEGImages
    │   ├── object_masks
    │   └── random_masks
    └── Train
        ├── JPEGImages
        ├── object_masks
        └── random_masks
```

## Train a model with FVI training set
```
cd src/
python train.py --config config.json --dataset_config dataset_configs/FVI_all_masks.json
```
Feel free to adjust parameters like batch_size and sample_length in these .json files.

- To train with FaceForensics dataset, please replace the `dataset_configs/FVI_all_masks.json` with `dataset_configs/forensics_all_masks.json`
- To use our implementation of baseline "Video Inpainting by Jointly Learning Temporal Structure and Spatial Detail", please replace `config.json` with `other_configs/3Dcomplete2D.json`

## Generate new free-form masks
### Generate mask and cluster them by masked ratio
```
# Generate masks with different parameters (e.g. number of stroke bound) in order to get diverse distribution of masked ratio
python scripts/mask_scripts/gen_masks.py \
    -vl 32 -nsb 1 20 -n 1000 \      # video length 32, number of strokes randomly in [1, 20], totally 1000 video masks generated
    --stroke_preset object_like \   # set the preset stroke type
    -od ./tmp/0_object_like \       # output directory
    --cluster_by_area \             # cluster the generated videos by areas into different directories (optional)
    --leave_boarder_unmasked 15     # after generation, create a copy with boarder unmasked for 15 pixels (optional)
```
You can add your own stroke preset in gen_masks.py to generate different styles of video mask strokes.
