ROOT_RESULT_DIR=$1
OUTPUT_FILENAME=$2
PREFIX="edge_connect_VOR_test_"
POSTFIX=".flist_results"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd  )"
VIDEO_DIR='/project/project-mira3/datasets/FreeFromVideoInpainting/test_20181109/JPEGImages'
ROOT_MASK_DIR="/project/project-mira3/datasets/FreeFromVideoInpainting/random_masks/test_set"
declare -a types=("object_like_middle_noBoarder" "object_like_middle" "rand_curve" "rand_curve_noBoarder")
declare -a percentages=("5" "15" "25" "35" "45" "55" "65" "uniform")

set -x
source activate FreeFormVideo1.0
cd $SCRIPT_DIR && cd ..

for t in "${types[@]}"
do
    for p in "${percentages[@]}"
    do
        MASK_DIR="$ROOT_MASK_DIR/$t/$p"
        RESULT_DIR="$ROOT_RESULT_DIR/$PREFIX${t}_${p}$POSTFIX"
        python evaluate.py -rgd $VIDEO_DIR -rmd $MASK_DIR -rrd $RESULT_DIR -o $OUTPUT_FILENAME
    done
done

