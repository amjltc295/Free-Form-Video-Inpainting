"""
This script is to generate VOR all-ratio-all-type mask test sets.
(All of the combination of different ratio and mask presets)
"""

import json


def get_loader(type, ratio):
    return {
        "type": "MaskedFrameDataLoader",
        "args": {
            "name": f"test_{type}_{ratio}",
            "root_videos_dir": "../dataset/test_20181109/JPEGImages",
            "root_masks_dir":
                f"../dataset/random_masks/test_set/{type}/{ratio}/",
            "root_outputs_dir": "../VOS_resized2",
            "dataset_args": {
                "type": "video",
                "w": 320,
                "h": 180,
                "sample_length": 100,
                "random_sample": False,
                "random_sample_mask": False,
                "mask_type": "from_start"
            },
            "batch_size": 1,
            "shuffle": False,
            "validation_split": 0.0,
            "num_workers": 4
        }
    }


def get_test_data_loaders(types, ratios):
    return [
        get_loader(type, ratio)
        for type in types for ratio in ratios
    ]


def save_json(name, dummy_train_data_loader, test_data_loaders):
    config = {
        "name": name,
        "data_loader": dummy_train_data_loader,
        "test_data_loader": test_data_loaders,
        "evaluate_test_warp_error": True
    }
    with open(f'./dataset_configs/{name}.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=False)


if __name__ == '__main__':
    dummy_train_data_loader = get_loader('rand_curve', '5')

    ratios = ['uniform'] + [str(i) for i in range(5, 70, 10)]
    test_data_loaders = get_test_data_loaders(
        ['rand_curve', 'object_like_middle', 'rand_curve_noBoarder', 'object_like_middle_noBoarder'],
        ratios
    )

    save_json('VOR_test_all_ratio_and_types', dummy_train_data_loader, test_data_loaders)
