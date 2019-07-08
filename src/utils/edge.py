import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # NOQA

import torch

from libs.CannyEdgePytorch.net_canny import Net as CannyEdgeNet


device = torch.device("cuda")
canny_edge_net = CannyEdgeNet(threshold=2.0, use_cuda=True).to(device)
canny_edge_net.eval()


def get_edge(tensor):
    with torch.no_grad():
        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = \
            canny_edge_net(tensor)
    return thresholded
