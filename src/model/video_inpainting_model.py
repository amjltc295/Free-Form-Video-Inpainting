# Based on https://github.com/avalonstrel/GatedConvolution_pytorch/
import torch

from base import BaseModel
from model.networks import (
    Generator,
    SNTemporalPatchGANDiscriminator
)


class VideoInpaintingModel(BaseModel):
    def __init__(self, opts, nc_in=5, nc_out=3, d_s_args={}, d_t_args={}):
        super().__init__()
        self.d_t_args = {
            "nf": 32,
            "use_sigmoid": True,
            "norm": 'SN'
        }  # default values
        for key, value in d_t_args.items():
            # overwrite default values if provided
            self.d_t_args[key] = value

        self.d_s_args = {
            "nf": 32,
            "use_sigmoid": True,
            "norm": 'SN'
        }  # default values
        for key, value in d_s_args.items():
            # overwrite default values if provided
            self.d_s_args[key] = value

        nf = opts['nf']
        norm = opts['norm']
        use_bias = opts['bias']

        # warning: if 2d convolution is used in generator, settings (e.g. stride,
        # kernal_size, padding) on the temporal axis will be discarded
        self.conv_by = opts['conv_by'] if 'conv_by' in opts else '3d'
        self.conv_type = opts['conv_type'] if 'conv_type' in opts else 'gated'

        self.use_refine = opts['use_refine'] if 'use_refine' in opts else False

        self.opts = opts

        ######################
        # Convolution layers #
        ######################
        self.generator = Generator(
            nc_in, nc_out, nf, use_bias, norm, self.conv_by, self.conv_type,
            use_refine=self.use_refine)

        #################
        # Discriminator #
        #################

        if 'spatial_discriminator' not in opts or opts['spatial_discriminator']:
            self.spatial_discriminator = SNTemporalPatchGANDiscriminator(
                nc_in=5, conv_type='2d', **self.d_s_args
            )
        if 'temporal_discriminator' not in opts or opts['temporal_discriminator']:
            self.temporal_discriminator = SNTemporalPatchGANDiscriminator(
                nc_in=5, **self.d_t_args
            )

    def forward(self, imgs, masks, guidances=None, model='G'):
        # imgs: [B, L, C=3, H, W]
        # masks: [B, L, C=1, H, W]
        # guidances: [B, L, C=1, H, W]
        if model == 'G':
            masked_imgs = imgs * masks
            output = self.generator(masked_imgs, masks, guidances)
        elif model == 'D_t':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances
            input_imgs = torch.cat([imgs, masks, guidances], dim=2)
            output = self.temporal_discriminator(input_imgs)
        elif model == 'D_s':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances
            input_imgs = torch.cat([imgs, masks, guidances], dim=2)
            # merge temporal dimension to batch dimension
            in_shape = list(input_imgs.shape)
            input_imgs = input_imgs.view([in_shape[0] * in_shape[1]] + in_shape[2:])
            output = self.spatial_discriminator(input_imgs)
            # split batch and temporal dimension
            output = output.view(in_shape[0], in_shape[1], -1)
        else:
            raise ValueError(f'forwarding model should be "G", "D_t", or "D_s", but got {model}')

        return output
