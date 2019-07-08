# codes are origined from https://github.com/Pika7ma/Temporal-Shift-Module/blob/master/tsm_util.py
import torch
import torch.nn.functional as F
from torch import nn


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, tensor):
        # not support higher order gradient
        # tensor = tensor.detach_()
        n, t, c, h, w = tensor.size()
        fold = c // 4
        ctx.fold_ = fold
        buffer_ = tensor.data.new(n, t, fold, h, w).zero_()
        buffer_[:, :-1] = tensor.data[:, 1:, :fold]
        tensor.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, 1:] = tensor.data[:, :-1, fold: 2 * fold]
        tensor.data[:, :, fold: 2 * fold] = buffer_
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer_ = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer_[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer_
        return grad_output, None


def tsm(tensor, version='zero', inplace=True):
    shape = B, T, C, H, W = tensor.shape
    split_size = C // 4
    if not inplace:
        pre_tensor, post_tensor, peri_tensor = tensor.split(
            [split_size, split_size, C - 2 * split_size],
            dim=2
        )
        if version == 'zero':
            pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]  # NOQA
            post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]  # NOQA
        elif version == 'circulant':
            pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],  # NOQA
                                     pre_tensor [:,   :-1, ...]), dim=1)  # NOQA
            post_tensor = torch.cat((post_tensor[:,  1:  , ...],  # NOQA
                                     post_tensor[:,   :1 , ...]), dim=1)  # NOQA
        else:
            raise ValueError('Unknown TSM version: {}'.format(version))
        return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(shape)
    else:
        out = InplaceShift.apply(tensor)
        return out


class TemporalShiftConv(nn.Module):
    def __init__(self, weights: list):
        super().__init__()
        shift_width = len(weights)
        weights = torch.tensor(weights).view(1, 1, shift_width, 1, 1)
        conv = nn.Conv3d(1, 1, [shift_width, 1, 1], bias=False, padding=[shift_width // 2, 0, 0])
        conv.weight = nn.Parameter(weights)
        self.conv = conv

    def forward(self, tensor):
        B, C, T, H, W = tensor.shape

        # For loop version:
        # channels = tensor.unbind(dim=1)  # unbind all channels to tuple
        # channels = [channel.unsqueeze(1) for channel in channels]  # restore the channel dimension
        # shifted = [self.conv(channel) for channel in channels]
        # shifted = torch.cat(shifted, dim=1)

        # Batch version:
        shifted = self.conv(tensor.contiguous().view([B * C, 1, T, H, W])).view([B, C, T, H, W])

        return shifted


class LearnableTSM(nn.Module):
    def __init__(self, shift_ratio=0.5, shift_groups=2, shift_width=3, fixed=False):
        super().__init__()
        self.shift_ratio = shift_ratio
        self.shift_groups = shift_groups
        if shift_groups == 2:  # for backward compability
            self.conv_names = ['pre_shift_conv', 'post_shift_conv']
        else:
            self.conv_names = [f'shift_conv_{i}' for i in range(shift_groups)]

        # shift kernel weights are initialized to behave like normal TSM
        pos = shift_width // 2
        back_shift_w = [0.] * shift_width
        back_shift_w[-pos] = 1.
        forward_shift_w = [0.] * shift_width
        forward_shift_w[pos - 1] = 1.
        for i in range(shift_groups):
            ts_conv = TemporalShiftConv(back_shift_w) if i * 2 < shift_groups else TemporalShiftConv(forward_shift_w)
            setattr(self, self.conv_names[i], ts_conv)

        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor):
        shape = B, C, T, H, W = tensor.shape
        split_size = int(C * self.shift_ratio) // self.shift_groups
        split_sizes = [split_size] * self.shift_groups + [C - split_size * self.shift_groups]
        tensors = tensor.split(split_sizes, dim=1)
        assert len(tensors) == self.shift_groups + 1

        tensors = [
            getattr(self, self.conv_names[i])(tensors[i])
            for i in range(self.shift_groups)
        ] + [tensors[-1]]
        return torch.cat(tensors, dim=1).view(shape)


# # It does not work as expected. Fix or abandon it.
# class FullyLearnableTSM(nn.Module):
#     def __init__(self, in_channels, shift_ratio=0.5):
#         super().__init__()
#         num_shift_channel = int(in_channels * shift_ratio)
#         self.split_sizes = [num_shift_channel, in_channels - num_shift_channel]
#         shift_conv = nn.Conv3d(
#             num_shift_channel, num_shift_channel, [3, 1, 1], stride=1,
#             padding=[1, 0, 0], groups=num_shift_channel, bias=False)
#         weight = self._get_shift_conv_init_weight(num_shift_channel)
#         shift_conv.weight = nn.Parameter(weight)
#         self.shift_conv = shift_conv

#     def _get_shift_conv_init_weight(self, num_shift_channel):
#         n = num_shift_channel // 2
#         m = num_shift_channel - n
#         weights_pre = torch.tensor([0., 0., 1.]).view(1, 1, 3, 1, 1).repeat(n, 1, 1, 1, 1)
#         weights_post = torch.tensor([1., 0., 0.]).view(1, 1, 3, 1, 1).repeat(m, 1, 1, 1, 1)
#         return torch.cat([weights_pre, weights_post], dim=0)

#     def forward(self, tensor):
#         shape = tensor.shape
#         tensors = list(tensor.split(self.split_sizes, dim=1))
#         tensors[0] = self.shift_conv(tensors[0])
#         return torch.cat(tensors, dim=1).view(shape)
