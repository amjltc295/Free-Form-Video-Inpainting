import torch
import torch.nn as nn

from base import BaseModel
from model.blocks import Conv3dBlock, Conv2dBlock


class VideoCN(nn.Module):
    def __init__(self, df=16):
        super().__init__()
        conv_kargs = {'norm': 'BN', 'activation': nn.ReLU(inplace=True)}
        conv_kargs_with_pad = {
            'norm': 'BN', 'activation': nn.ReLU(inplace=True), 'padding': -1}

        self.conv1 = Conv3dBlock(4, df, 5, **conv_kargs_with_pad)
        self.conv2 = Conv3dBlock(df, df * 2, 3, stride=(1, 2, 2), **conv_kargs_with_pad)
        self.conv3 = Conv3dBlock(df * 2, df * 4, 3, **conv_kargs_with_pad)
        self.conv4 = Conv3dBlock(df * 4, df * 8, 3, stride=(1, 2, 2), **conv_kargs_with_pad)

        self.conv5 = Conv3dBlock(df * 8, df * 16, 3, dilation=(1, 2, 2), **conv_kargs_with_pad)
        self.conv6 = Conv3dBlock(df * 16, df * 16, 3, dilation=(1, 4, 4), **conv_kargs_with_pad)
        self.conv7 = Conv3dBlock(df * 16, df * 16, 3, dilation=(1, 8, 8), **conv_kargs_with_pad)

        self.conv8 = Conv3dBlock(df * 16, df * 8, 3, **conv_kargs_with_pad)
        self.conv9 = Conv3dBlock(
            df * 8 * 2, df * 4, 4, stride=(1, 2, 2), padding=(1, 1, 1),
            transpose=True, **conv_kargs)

        self.conv10 = Conv3dBlock(df * 4 * 2, df * 2, 3, **conv_kargs_with_pad)
        self.conv11 = Conv3dBlock(
            df * 2 * 2, df, 4, stride=(1, 2, 2), padding=(1, 1, 1),
            transpose=True, **conv_kargs)

        self.conv12 = Conv3dBlock(df, 3, 3, norm=None, activation=None, padding=-1)

    def _cut_to_align(self, x, y):
        """
            Cut the boarder by 1 frame(temporal)/row(height)/col(width)
            such that dimensions of x and y could be the same.
        """
        output_x = x
        # dimension: [batch, channel, temporal, width, height]
        # target 2, 3, 4  ->             ^^^     ^^^    ^^^
        for i in range(2, 5):
            dim_x = x.shape[i]
            dim_y = y.shape[i]
            if dim_x != dim_y:
                assert dim_x == dim_y + 1, ('Only deal with the odd width inverse case of deconv. ',
                                            f'Got dim_x: {dim_x}, dim_y: {dim_y}')
                output_x = output_x.narrow(i, 0, dim_y)
        return output_x

    def forward(self, x):
        # print(f'input size: {x.shape}')
        intermediate = {}
        # Conv 1 ~ 8
        for i in range(1, 9):
            x = getattr(self, f'conv{i}')(x)
            intermediate[i] = x
            # print(f'intermediate[{i}] size: {x.shape}')

        x = torch.cat([x, intermediate[4]], dim=1)
        x = self.conv9(x)
        x = self._cut_to_align(x, intermediate[3])
        # print(f'intermediate[9] size: {x.shape}')

        x = torch.cat([x, intermediate[3]], dim=1)
        x = self.conv10(x)
        # print(f'intermediate[10] size: {x.shape}')

        x = torch.cat([x, intermediate[2]], dim=1)
        x = self.conv11(x)
        x = self._cut_to_align(x, intermediate[1])
        # print(f'intermediate[11] size: {x.shape}')
        imcomplete_video = self.conv12(x)
        # print(f'imcomplete_video size: {imcomplete_video.shape}')

        return imcomplete_video


class CombCN(nn.Module):
    def __init__(self, df=32):
        super().__init__()
        conv_kargs_with_pad = {
            'norm': 'BN', 'activation': nn.ReLU(inplace=True), 'padding': -1}

        self.conv1 = Conv2dBlock(4, df * 2, 5, **conv_kargs_with_pad)
        self.conv2 = Conv2dBlock(df * 2, df * 4, 3, stride=2, **conv_kargs_with_pad)
        self.conv3 = Conv2dBlock(df * 4 + 3, df * 4, 3, **conv_kargs_with_pad)
        self.conv4 = Conv2dBlock(df * 4, df * 8, 3, stride=2, **conv_kargs_with_pad)
        self.conv5 = Conv2dBlock(df * 8, df * 8, 3, **conv_kargs_with_pad)
        self.conv6 = Conv2dBlock(df * 8, df * 8, 3, **conv_kargs_with_pad)

        self.conv7 = Conv2dBlock(df * 8, df * 8, 3, dilation=2, **conv_kargs_with_pad)
        self.conv8 = Conv2dBlock(df * 8, df * 8, 3, dilation=4, **conv_kargs_with_pad)
        self.conv9 = Conv2dBlock(df * 8, df * 8, 3, dilation=8, **conv_kargs_with_pad)
        self.conv10 = Conv2dBlock(df * 8 * 2, df * 8, 3, dilation=16, **conv_kargs_with_pad)

        self.conv11 = Conv2dBlock(df * 8 * 2, df * 8, 3, **conv_kargs_with_pad)
        self.conv12 = Conv2dBlock(df * 8 * 2, df * 8, 3, **conv_kargs_with_pad)
        self.conv13 = Conv2dBlock(
            df * 8 * 2, df * 4, 4, stride=2,
            transpose=True, **conv_kargs_with_pad)

        self.conv14 = Conv2dBlock(df * 4 * 2, df * 4, 3, **conv_kargs_with_pad)
        self.conv15 = Conv2dBlock(
            df * 4 * 2 + 3, df * 2, 4, stride=2,
            transpose=True, **conv_kargs_with_pad)

        self.conv16 = Conv2dBlock(df * 2 * 2, df, 3, **conv_kargs_with_pad)
        self.conv17 = Conv2dBlock(df, 3, 3, norm=None, activation=None, padding=-1)

    def forward(self, x, imcomplete_frame):
        # print(f'input size: {x.shape}')
        intermediate = {}
        # Conv 1 ~ 9
        for i in range(1, 10):
            if i == 3:
                x = torch.cat([x, imcomplete_frame], dim=1)

            x = getattr(self, f'conv{i}')(x)
            intermediate[i] = x
            # print(f'intermediate[{i}] size: {x.shape}')

        # Conv 10 ~ 16
        for i in range(10, 17):
            if i == 15:
                x = torch.cat([x, imcomplete_frame], dim=1)

            j = 17 - i
            x = torch.cat([x, intermediate[j]], dim=1)
            x = getattr(self, f'conv{i}')(x)
            # print(f'intermediate[{i}] size: {x.shape}')

        completed_frame = self.conv17(x)
        # print(f'completed_frame size: {completed_frame.shape}')
        return completed_frame


class CompletionNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.videoCN = VideoCN()
        self.combCN = CombCN()

    def forward(self, imgs, masks, guidances=None):
        masked_imgs = imgs * masks

        # B, L, C, H, W -> B, C, L, H, W
        masked_video = masked_imgs.transpose(1, 2)
        masks = masks.transpose(1, 2)

        d_masks = nn.functional.interpolate(masks, scale_factor=[1, 0.5, 0.5])
        d_masked_video = nn.functional.interpolate(masked_video, scale_factor=[1, 0.5, 0.5])

        d_input = torch.cat([d_masked_video, d_masks], dim=1)

        imcomplete_video = self.videoCN(d_input)
        imcomplete_video = imcomplete_video * (1 - d_masks) + d_masked_video * d_masks

        output = []
        for i in range(imcomplete_video.shape[2]):
            masked_frame = masked_video[:, :, i]
            mask = masks[:, :, i]
            imcomplete_frame = imcomplete_video[:, :, i]

            comb_input = torch.cat([masked_frame, mask], dim=1)
            completed_frame = self.combCN(comb_input, imcomplete_frame)
            output.append(completed_frame)
        output = torch.stack(output, dim=2)

        output = (1 - masks) * output + masks * masked_video
        output = output.transpose(1, 2)

        model_output = {
            "outputs": output,
            "imcomplete_video": imcomplete_video
        }

        # print(f'Final output size: {output.shape}')
        return model_output
