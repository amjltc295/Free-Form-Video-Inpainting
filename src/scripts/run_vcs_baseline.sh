set -x  # echo on
ROOT_MASK_DIR="/project/project-mira3/datasets/FreeFromVideoInpainting/random_masks/test_set"
ROOT_MASK_VIDEO_DIR="/project/project-mira3/datasets/FreeFromVideoInpainting/random_masks_videos/test_set"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd  )"
MASK_TYPE=$1
VCS_DIR=$2
VIDEO_DIR='/project/project-mira3/datasets/FreeFromVideoInpainting/test_20181109_videos'
TARGET_MASK_DIR="$ROOT_MASK_DIR/$MASK_TYPE"
declare -a percentages=("5" "15" "25" "35" "45" "55" "65" "uniform")

echo "TARGET_MASK_DIR: $TARGET_MASK_DIR"
echo "ROOT_MASK_VIDEO_DIR: $ROOT_MASK_VIDEO_DIR"
echo "VCS_DIR: $VCS_DIR"
echo "VIDEO_DIR: $VIDEO_DIR"
echo ""

# Ask for consent
read -p "Are you sure ([N]/Yy)? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$  ]]
then

    source activate FreeFormVideo1.0
    # Prepare video
    rm -rf $VCS_DIR/dataset/video
    cp -r $VIDEO_DIR $VCS_DIR/dataset/video

    # Run for each percentage of mask
    for p in "${percentages[@]}"
    do
        MASK_DIR="$ROOT_MASK_DIR/${MASK_TYPE}/$p"
        MASK_VIDEO_DIR="$ROOT_MASK_VIDEO_DIR/$MASK_TYPE/$p"
        echo "MASK_DIR: $MASK_DIR \nMASK_VIDEO_DIR: $MASK_VIDEO_DIR" >> vsc_record.txt

        # Prepare mask video
        rm -rf $MASK_VIDEO_DIR

        cd $SCRIPT_DIR
        python gen_output_videos.py -id $MASK_DIR -od $MASK_VIDEO_DIR -ml 15
        python rename.py -d $MASK_VIDEO_DIR -t file -pf _hole

        cd $VCS_DIR
        mv $VCS_DIR/dataset/hole $VCS_DIR/dataset/${MASK_TYPE}_${p}_masks
        cp -r $MASK_VIDEO_DIR ./dataset/hole

        # Start running
        git checkout VOR
        LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /auto/matlab-2018b/bin/matlab -nodisplay -nosplash -nodesktop -r "try, run('$VCS_DIR/run_all.m'), catch, exit(1), end, exit(0);"

        # Rename output folder
        mv $VCS_DIR/result $VCS_DIR/vcs_results_${MASK_TYPE}_${p}
    done

fi
