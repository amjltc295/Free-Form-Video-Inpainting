import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # NOQA

import torch
import torch.nn as nn
# import numpy as np

from model.vgg import Vgg16


device = torch.device("cuda")
vgg = Vgg16(requires_grad=False).to(device)


class ReconLoss(nn.Module):
    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        if self.masked:
            masks = data_input['masks']
            return self.loss_fn(outputs * (1 - masks), targets * (1 - masks))
        else:
            return self.loss_fn(outputs, targets)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def vgg_loss(self, output, target):
        output_feature = vgg(output)
        target_feature = vgg(target)
        loss = (
            self.l1_loss(output_feature.relu2_2, target_feature.relu2_2)
            + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3)
            + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        )
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(
                self.vgg_loss(outputs[:, frame_idx], targets[:, frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class StyleLoss(nn.Module):
    def __init__(self, original_channel_norm=True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.original_channel_norm = original_channel_norm

    # From https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    # Implement "Image Inpainting for Irregular Holes Using Partial Convolutions", Liu et al., 2018
    def style_loss(self, output, target):
        output_features = vgg(output)
        target_features = vgg(target)
        layers = ['relu2_2', 'relu3_3', 'relu4_3']  # n_channel: 128 (=2 ** 7), 256 (=2 ** 8), 512 (=2 ** 9)
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)  # original design (avoid too small loss)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(
                self.style_loss(outputs[:, frame_idx], targets[:, frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def edge_loss(self, output, target):
        from utils.edge import get_edge
        output_edge = get_edge(output)
        gt_edge = get_edge(target)
        loss = self.l1_loss(output_edge, gt_edge)
        return loss, output_edge, gt_edge

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        mean_image_loss = []
        output_edges = []
        target_edges = []
        for batch_idx in range(targets.size(0)):
            edges_o = []
            edges_t = []
            for frame_idx in range(targets.size(1)):
                loss, output_edge, target_edge = self.edge_loss(
                    outputs[batch_idx, frame_idx:frame_idx + 1],
                    targets[batch_idx, frame_idx:frame_idx + 1]
                )
                mean_image_loss.append(loss)
                edges_o.append(output_edge)
                edges_t.append(target_edge)
            output_edges.append(torch.cat(edges_o, dim=0))
            target_edges.append(torch.cat(edges_t, dim=0))

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        self.current_output_edges = output_edges
        self.current_target_edges = target_edges
        return mean_image_loss


class L1LossMaskedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, mask):
        masked = 1 - mask
        l1_sum = self.l1(x * masked, y * masked)
        return l1_sum / torch.sum(masked)


class L2LossMaskedMean(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.l2 = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, mask):
        masked = 1 - mask
        l2_sum = self.l2(x * masked, y * masked)
        return l2_sum / torch.sum(masked)


class ImcompleteVideoReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        imcomplete_video = model_output['imcomplete_video']
        targets = data_input['targets']
        down_sampled_targets = nn.functional.interpolate(
            targets.transpose(1, 2), scale_factor=[1, 0.5, 0.5])

        masks = data_input['masks']
        down_sampled_masks = nn.functional.interpolate(
            masks.transpose(1, 2), scale_factor=[1, 0.5, 0.5])
        return self.loss_fn(
            imcomplete_video, down_sampled_targets,
            down_sampled_masks
        )


class CompleteFramesReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        masks = data_input['masks']
        return self.loss_fn(outputs, targets, masks)


# From https://github.com/phoenix104104/fast_blind_video_consistency
class TemporalWarpingLoss(nn.Module):
    def __init__(self, flownet_checkpoint_path=None, alpha=50):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()
        self.alpha = alpha

        self.flownet_checkpoint_path = flownet_checkpoint_path
        self.flownet = None

    def get_flownet_checkpoint_path(self):
        return self.flownet_checkpoint_path

    def _setup(self):
        from libs.flownet2_pytorch.flownet_wrapper import FlowNetWrapper
        self.flownet = FlowNetWrapper(checkpoint_path=self.flownet_checkpoint_path)

    def _get_non_occlusion_mask(self, targets, warped_targets):
        non_occlusion_masks = torch.exp(
            -self.alpha * torch.sum(
                targets[:, 1:] - warped_targets, dim=2
            ).pow(2)
        ).unsqueeze(2)
        return non_occlusion_masks

    def _get_loss(self, outputs, warped_outputs, non_occlusion_masks, masks):
        return self.loss_fn(
            outputs[:, 1:] * non_occlusion_masks,
            warped_outputs * non_occlusion_masks,
            masks[:, 1:]
        )

    def forward(self, data_input, model_output):
        if self.flownet is None:
            self._setup()

        targets = data_input['targets'].to(device)
        outputs = model_output['outputs'].to(device)
        flows = self.flownet.infer_video(targets).to(device)

        from utils.flow_utils import warp_optical_flow
        warped_targets = warp_optical_flow(targets[:, :-1], -flows).detach()
        warped_outputs = warp_optical_flow(outputs[:, :-1], -flows).detach()
        non_occlusion_masks = self._get_non_occlusion_mask(targets, warped_targets)

        # model_output is passed by name and dictionary is mutable
        # These values are sent to trainer for visualization
        model_output['warped_outputs'] = warped_outputs[0]
        model_output['warped_targets'] = warped_targets[0]
        model_output['non_occlusion_masks'] = non_occlusion_masks[0]
        from utils.flow_utils import flow_to_image
        flow_imgs = []
        for flow in flows[0]:
            flow_img = flow_to_image(flow.cpu().permute(1, 2, 0).detach().numpy()).transpose(2, 0, 1)
            flow_imgs.append(torch.Tensor(flow_img))
        model_output['flow_imgs'] = flow_imgs

        masks = data_input['masks'].to(device)
        return self._get_loss(outputs, warped_outputs, non_occlusion_masks, masks)


class TemporalWarpingError(TemporalWarpingLoss):
    def __init__(self, flownet_checkpoint_path, alpha=50):
        super().__init__(flownet_checkpoint_path, alpha)
        self.loss_fn = L2LossMaskedMean(reduction='none')

    def _get_loss(self, outputs, warped_outputs, non_occlusion_masks, masks):
        # See https://arxiv.org/pdf/1808.00449.pdf 4.3
        # The sum of non_occlusion_masks is different for each video,
        # So the batch dim is kept
        loss = self.loss_fn(
            outputs[:, 1:] * non_occlusion_masks,
            warped_outputs * non_occlusion_masks,
            masks[:, 1:]
        ).sum(1).sum(1).sum(1).sum(1)

        loss = loss / non_occlusion_masks.sum(1).sum(1).sum(1).sum(1)
        return loss.sum()


# From https://github.com/jxgu1016/Total_Variation_Loss.pytorch
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data_input, model_output):
        # View 3D data as 2D
        outputs = model_output['outputs']
        B, L, C, H, W = outputs.shape
        x = outputs.view([B * L, C, H, W])

        masks = data_input['masks']
        masks = masks.view([B * L, -1])
        mask_areas = masks.sum(dim=1)

        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum(1).sum(1).sum(1)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum(1).sum(1).sum(1)
        return ((h_tv + w_tv) / mask_areas).mean()


# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

        elif type == 'l1':
            self.criterion = nn.L1Loss()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
