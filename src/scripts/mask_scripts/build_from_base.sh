# Please run it by `zsh scripts/mask_scripts/build_from_base.sh` under `src/`
for f in ../dataset/random_masks/base/*
do
    name=$(basename $f)
    python scripts/mask_scripts/copy_from_base.py --base ../dataset/random_masks/base/${name} --name ${name} --dst ../dataset/random_masks/test_set/${name}/uniform --start 100 --num 17
done

for ratio in {5,15,25,35,45,55};
do
    for f in ../dataset/random_masks/base/*
    do
        name=$(basename $f)
        python scripts/mask_scripts/copy_from_base.py --base ../dataset/random_masks/base/${name} --name ${name} --dst ../dataset/random_masks/test_set/${name}/${ratio} --start 0 --num 100 --min_area ${ratio} --max_area ${ratio}
    done
done

