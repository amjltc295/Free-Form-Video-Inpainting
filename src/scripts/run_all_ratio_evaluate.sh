set -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"   )" >/dev/null 2>&1 && pwd   )"
source activate FreeFormVideo1.0

GD="../dataset/test_20181109/JPEGImages"
RMD="../dataset/random_masks/test_set"
RRD=$1
OF=$2
touch ${OF}

for type_dir in $RMD/*
do
    type=$(basename $type_dir)
    for ratio in {uniform,5,15,25,35,45,55,65}
    do
        MD=${RMD}/${type}/${ratio}
        RD=${RRD}/test_${type}_${ratio}
        if [ ! -d "$RD"  ]; then
            continue
        fi
        echo type: $type ratio: $ratio | tee -a $OF
        CUDA_VISIBLE_DEVICES=1 python evaluate.py -rgd $GD -rrd $RD -rmd $MD --verbose 0 | tee -a $OF
        echo "" | tee -a $OF
    done
    echo "" | tee -a $OF
done

