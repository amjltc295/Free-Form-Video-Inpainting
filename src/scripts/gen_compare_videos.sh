# The relative directory structure by default follows the ones in Liu's lab desktop
set -x

# # Random masks on VOR
# RGD=../dataset/test_20181109/JPEGImages
# for type in {object_like_middle,object_like_middle_noBoarder,rand_curve,rand_curve_noBoarder}
# do
#     for ratio in {uniform,5,15,25,35,45,55,65}
#     do
#         RMD=../dataset/testset_with_mask/${type}/${ratio}
#         R_TCCDS=../all_outputs/2016Huang/vcs_results_${type}_${ratio}/completion_ours
#         R_EC=../all_outputs/edge-connect_not_finetuned/edge_connect_VOR_test_${type}_${ratio}.flist_results
#         R_3D2D=../all_outputs/VOR_test_all_ratio_3D2D_0225_202152_e300/epoch_0/test_${type}_${ratio}
#         R_3DGated=../all_outputs/VOR_test_all_ratio_ours_0211_0242_e350_p0.1209.pth/epoch_0/test_${type}_${ratio}
#         R_OURS= ../all_outputs/LGTSM_width3_GANnf64_firstTSMoff_0424_221151_e99/epoch_0/test_${type}_${ratio}
#         python scripts/compare_results.py -vs $RMD $R_TCCDS $R_EC $R_3D2D $R_3DGated $R_OURS -names "Mask" "TCCDS" "Edge Connect" "CombCN" "3D Gated" "LGTSM" -fnwrl 1 3 1 -od ./compare_LGTSM/${type}/${ratio} --assume_ordered --col 3 --name_prefix ${type}_${ratio}_
#     done
# done

# Forensics
# RGD=../dataset/forensics/test_32frames
# for type in {test_object_like,test_rand_curve,test_bbox}
# do
#     RMD=../dataset/forensics/masked_input/${type}
#     R_TCCDS=../forensics_all_outputs/TCCDS_FaceForensics_results/vcs_FaceForensics_results_${type}/completion_ours
#     # R_EC=../forensics_all_outputs/edge_connect_FaceForensics_results/edge_connect_2stage_FaceForensics_${type}.flist_results/
#     R_EC=../forensics_all_outputs/EC_from_scratch/edge_connect_from_scratch_FaceForensics_${type}.flist_results/
#     R_3D2D=../forensics_all_outputs/3D2D_decay_e140/${type}
#     R_3DGated=../forensics_all_outputs/ours/${type}
#     R_OURS=../forensics_all_outputs/face_GatedTSM_inplace_noskip_b2_back_L1_vgg_style_TSMSNTPD_1_1_10_1_load300/${type}
#     set -x; python scripts/compare_results.py -vs $RMD $R_TCCDS $R_EC $R_3D2D $R_3DGated $R_OURS -names "Mask" "TCCDS" "Edge Connect" "CombCN" "3D Gated" "Ours" -fnwrl 1 2 1 -od ../forensics_all_outputs/compare_vidoes_LGTSM/${type} --assume_ordered --col 3 --name_prefix ${type}_ ; set +x
# done

# Ablation study
# RGD=../dataset/test_20181109/JPEGImages
# for type in {object_like_middle,rand_curve}
# do
#     ratio=uniform
#     RMD=../dataset/testset_with_mask/${type}/${ratio}
#     R_OURS=../all_outputs/VOR_test_all_ratio_ours_0211_0242_e350_p0.1209.pth/epoch_0/test_${type}_${ratio}
#     R_NOTPD=../all_outputs/ablation/output/noTPD_e120/epoch_0/test_${type}_${ratio}/
#     R_VANILLA=../all_outputs/ablation/output/vanilla/epoch_0/test_${type}_${ratio}/
#     R_2D=../all_outputs/ablation/output/2d/epoch_0/test_${type}_${ratio}/
#     python scripts/compare_results.py -vs $RMD $R_2D $R_VANILLA $R_NOTPD $R_OURS $RGD -names "Mask" "W/o 3D Conv" "W/o Gated Conv" "W/o T-PatchGAN" "Our Full Model" "Ground Truth" -fnwrl 1 3 1 -od ../all_outputs/ablation/compare_videos/${type}/${ratio} --assume_ordered --col 3 --name_prefix ${type}_${ratio}_
# done

# # Object removal
# RGD=../dataset/test_20181109/JPEGImages
# R_TCCDS=./output/object_removal/vcs_VOR_VORmask_results/completion_ours
# R_EC=./output/object_removal/outputs_edge_connect_VOR_VORmask_results
# R_3D2D=./output/object_removal/CombCN/epoch_0/test_object_removal_dilate_5
# R_3DGated=./output/object_removal/ours/ours_dilated5/epoch_0/test_object_removal_dilate_5
# R_OURS=./output/object_removal/ours/????/epoch_0/test_object_removal_dilate_5
# python scripts/compare_results.py -vs $R_TCCDS $R_EC $R_3D2D $R_3DGated $R_OURS -names "TCCDS" "Edge Connect" "CombCN" "Ours" "Original" -fnwrl 1 3 1 -od ./output/object_removal/compare_videos_LGTSM --assume_ordered --col 3

# # Subscript removal
# target=ours
# RGD=../dataset/test_20181109/JPEGImages
# RMD=./output/subscript_slow/${target}/epoch_0/test_special_icon/pasted_inputs
# R_OURS=./output/subscript_slow/${target}/epoch_0/test_special_icon/results
# python scripts/compare_results.py -vs $RMD $R_OURS $RGD -names "Input" "Result" "Ground Truth" -fnwrl 1 3 1 -od ./output/subscript_slow/${target}/compare_videos --assume_ordered --col 3

