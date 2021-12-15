from torch.nn import functional as F
import torch.nn as nn
import torch


class Projection_network(nn.Module):
    def __init__(self, nc=80, output_node = 128, ch= (192, 384, 768)):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = 3
        self.na = 3 # number of anchors
        self.output_node = output_node
        self.m = nn.ModuleList(nn.Conv2d(x, self.na * self.output_node, 1) for x in ch)  # output conv
    def forward(self, x):
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = F.normalize(x[i]) # do normalization
            x[i] = self.m[i](x[i])  # conv
            x[i] = F.normalize(x[i])
            x[i] = x[i].view(bs, self.na, self.output_node, ny, nx).permute(0, 1, 3, 4, 2).contiguous() 
        return x

class Compute_sup_loss:
    def __init__(self, build_target=None, temperature = 3.0) -> None:
        self.build_targets = build_target
        self.temperature = temperature
    def __call__(self, pred, targets):
        tcls, tbox, indices, anchors = self.build_targets(pred, targets)
        X = []
        Y = []
        for i, z in enumerate(pred):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            zs = z[b, a, gj, gi]
            X.append(zs)
            Y.append(tcls[i])
        X = torch.cat(X)
        Y = torch.cat(Y)
        exp_dot_result = torch.exp(torch.matmul(X, X.T) / self.temperature)

        denominator = torch.sum(exp_dot_result, dim=1)
        sup_loss = torch.zeros(1).cuda()
        for data_idx ,cat_id in enumerate(Y):
            pos = (Y == cat_id) # positive
            pos[data_idx] = False # remove itself
            loss = torch.log(exp_dot_result[data_idx, pos] / (denominator[data_idx] - exp_dot_result[data_idx, data_idx])).sum()
            sup_loss += loss / pos.sum()
        sup_loss *= -1
        sup_loss /= X.shape[0]
        return sup_loss