# Script Usage
## Generate n-result comparison videos
```
# Generate Input vs Output comparison
python scripts/compare_results.py -vs v0.1.5_e260_input v0.1.5_e260_out -od v0.1.5_e260_compare -names Input Ours

# Generate Input vs Output comparison, repeating 3 times with 1x, 3x, 1x speed
python scripts/compare_results.py -vs v0.1.5_e260_input v0.1.5_e260_out -od v0.1.5_e260_compare -names Input Ours -fnwrl 1 3 1
```

## Clean saved experiments example
```
python scripts/clean_saved.py -p "*v0.3.0*" --at_least 10 -all
```

## Extract frames from YouTube-BB videos
```
python scripts/extract_frames.py -od ../dataset/youtube-bb_images_20190224_t -rid /project/project-mira3/datasets/youtube-bb/yt_bb_detection_train -min_w 640 -min_h 360 -n 16
```

## Run baseline: Temporally-Coherent-Completion-of-Dynamic-Video
```
# Run different mask type manually
# bash scripts/run_vcs_baseline.sh <mask type> <forked base directory>
# Default ROOT_MASK_DIR="/project/project-mira3/datasets/FreeFromVideoInpainting/random_masks/test_set"                                                                         
# Default ROOT_MASK_VIDEO_DIR="/project/project-mira3/datasets/FreeFromVideoInpainting/random_masks_videos/test_set" 
# For example:
bash scripts/run_vcs_baseline.sh object_like /tmp2/yaliangchang/Temporally-Coherent-Completion-of-Dynamic-Video

```

## An example to evaluate distance between output and ground truth videos
```
python evaluate.py -rgd ../dataset/test_20181109/JPEGImages -rmd ../dataset/random_masks_vl20_ns5_object_like_test/ -rrd saved/VideoInpaintingModel_v0.3.0_l1_m+a_mvgg_style_1_6_0.05_120_0_0_all_mask/0102_214744/test_outputs

```

### Evaluate baseline: Edge-Connect
```
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_dirs.sh <root_output_dir, e.g., /tmp2/yaliangchang/edge-connect/outputs>  <output filename, e.g., edge_connect_result.txt>
```
