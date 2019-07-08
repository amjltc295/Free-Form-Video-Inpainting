# The relative directory structure by default follows the ones in Liu's lab desktop

RGD=../dataset/test_20181109/JPEGImages
for type in {object_like_middle,object_like_middle_noBoarder,rand_curve,rand_curve_noBoarder}
do
    for ratio in {uniform,5,15,25,35,45,55,65}
    do
        RMD=../dataset/testset_with_mask/${type}/${ratio}
        R_TCCDS=../all_outputs/2016Huang/vcs_results_${type}_${ratio}/completion_ours
        R_EC=../all_outputs/edge-connect/edge_connect_VOR_test_${type}_${ratio}.flist_results
        R_3D2D=../all_outputs/VOR_test_all_ratio_3D2D_0225_202152_e300/epoch_0/test_${type}_${ratio}
        R_OURS=../all_outputs/VOR_test_all_ratio_ours_0211_0242_e350_p0.1209.pth/epoch_0/test_${type}_${ratio}
        python scripts/compare_results.py -vs $RGD $RMD $R_TCCDS $R_EC $R_3D2D $R_OURS -names "Ground Truth" "Mask" "TCCDS" "Edge Connect" "CombCN" "Ours" -fnwrl 1 3 1 -od ./compare/${type}/${ratio} --assume_ordered --col 3
    done
done
