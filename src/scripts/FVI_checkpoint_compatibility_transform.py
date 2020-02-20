import re
import argparse
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        """
        IMPORTANT NOTE: to run this script, you need to put it under src/ instead of src/scripts
        due to some unpickling issues of torch.load()

        This script is to modify a checkpoint saved in our previous "FVI with Gated Conv" repository.
        What it will do:
            1. Add a "use_skip_connection=True" tag in the "arch/args/opts" entry in the config json.
            2. Replace names of nn modules that has been changed during the LGTSM development.
        """
    )
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument(
        '--add_course_net_structures', action='store_true',
        help='For those older checkpoint, there\'s no "Generator.CourseNet" structures. '
             'Set this arg to add it manually.')
    args = parser.parse_args()
    return args


def rename_conv_to_featureConv(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict([
        (k.replace('.conv.', '.featureConv.'), v)
        if 'temporal_discriminator' in k and '.conv.' in k else (k, v)
        for k, v in state_dict.items()
    ])
    new_metadata = OrderedDict([
        (re.sub(r'.conv$', '.featureConv', k), v)
        if 'temporal_discriminator' in k and '.conv' in k else (k, v)
        for k, v in state_dict._metadata.items()
    ])
    setattr(new_state_dict, '_metadata', new_metadata)
    checkpoint['state_dict'] = new_state_dict
    return checkpoint


def add_course_net_structures(checkpoint):
    state_dict = checkpoint['state_dict']
    downsample_conv_names = [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6',
        'dilated_conv1', 'dilated_conv2', 'dilated_conv3', 'dilated_conv4',
        'conv7', 'conv8'
    ]
    upsample_conv_names = [
        'conv9', 'conv10', 'conv11', 'deconv1', 'deconv2'
    ]

    def transform(k, v):
        if any([k.startswith(name) for name in upsample_conv_names]):
            out = ('generator.coarse_net.upsample_module.' + k, v)
        elif any([k.startswith(name) for name in downsample_conv_names]):
            out = ('generator.coarse_net.downsample_module.' + k, v)
        else:
            out = (k, v)
        return out

    new_state_dict = OrderedDict([
        transform(k, v)
        for k, v in state_dict.items()
    ])
    new_metadata = OrderedDict([
        transform(k, v)
        for k, v in state_dict._metadata.items()
    ])
    setattr(new_state_dict, '_metadata', new_metadata)
    checkpoint['state_dict'] = new_state_dict
    return checkpoint


def main():
    args = parse_args()
    print(f'\nLoading the old checkpoint from {args.src}')
    checkpoint = torch.load(args.src)

    print('\nRenaming state_dict keys and adding the skip connection tag')
    new_checkpoint = rename_conv_to_featureConv(checkpoint)
    new_checkpoint['config']['arch']['args']['opts']['use_skip_connection'] = True

    if args.add_course_net_structures:
        print('\nAdding course_net structures into checkpoint')
        new_checkpoint = add_course_net_structures(new_checkpoint)

    print(f'\nSaving the modified checkpoint into {args.dst}')
    torch.save(new_checkpoint, args.dst)
    print('\nDone')


if __name__ == '__main__':
    main()
