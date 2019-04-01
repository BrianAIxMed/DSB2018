import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)

class MeanSquareErrorLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.mse_loss = nn.MSELoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.mse_loss(inputs, targets)

# DICE = 2 * Sum(PiGi) / (Sum(Pi) + Sum(Gi))
# Refer https://github.com/pytorch/pytorch/issues/1249 for Laplace/Additive smooth
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice = score.sum() / num
        # three kinds of loss formulas: (1) 1 - dice (2) -dice (3) -torch.log(dice)
        return 1. - dice


# Jaccard/IoU = Sum(PiGi) / (Sum(Pi) + Sum(Gi) - Sum(PiGi))
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1. - iou

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection [https://arxiv.org/abs/1708.02002]
    Digest the paper as below:
        α, balances the importance of positive/negative examples
        γ, focusing parameter that controls the strength of the modulating term

            CE(pt) = −log(pt) ==> pt = exp(-CE)
            FL(pt) = −α((1 − pt)^γ) * log(pt)

        In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).
    """
    def __init__(self, focusing_param=2, balance_param=0.25):
        super().__init__()
        self.gamma = focusing_param
        self.alpha = balance_param

    def forward(self, inputs, targets, weights=None):
        logpt = -binary_cross_entropy(inputs, targets, weights)
        pt = torch.exp(logpt)
        # compute the loss
        focal_loss = -((1-pt)**self.gamma) * logpt
        balanced_focal_loss = self.alpha * focal_loss
        return balanced_focal_loss

class CORAL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, weights=None):
        Ci = self.corrcoef(inputs, inputs)
        Co = self.corrcoef(targets, targets)
        return torch.norm(Ci - Co, 2)

    def corrcoef(self, x, y):  # https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/2
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

class MMD():
    # copied from https://github.com/JorisRoels/domain-adaptive-segmentation/blob/master/networks/mmd.py
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.GAMMA = 10 ^ 3

    def forward(self, source, target):

        K_XX, K_XY, K_YY, d = self.mix_rbf_kernel(source, target, [self.GAMMA])

        return self.mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True)

    def mix_rbf_kernel(self, X, Y, sigma_list):
        assert (X.size(0) == Y.size(0))
        m = X.size(0)

        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma ** 2)
            K += torch.exp(-gamma * exponent)

        return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

    def mmd2(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = K_XX.size(0)  # assume X, Y are same shape

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if const_diagonal is not False:
            diag_X = diag_Y = const_diagonal
            sum_diag_X = sum_diag_Y = m * const_diagonal
        else:
            diag_X = torch.diag(K_XX)  # (m,)
            diag_Y = torch.diag(K_YY)  # (m,)
            sum_diag_X = torch.sum(diag_X)
            sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

        if biased:
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2.0 * K_XY_sum / (m * m))
        else:
            mmd2 = (Kt_XX_sum / (m * (m - 1))
                    + Kt_YY_sum / (m * (m - 1))
                    - 2.0 * K_XY_sum / (m * m))

        return mmd2

def criterion(preds, labels):
# (1) BCE Loss
#     return BinaryCrossEntropyLoss2d().forward(preds, labels)
# (2) BCE Loss + DICE Loss
#     return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
#            SoftDiceLoss().forward(preds, labels)
# (3) BCE Loss + Jaccard/IoU Loss
    return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
           IoULoss().forward(preds, labels)

def mse_criterion(preds, labels):
    return MeanSquareErrorLoss2d().forward(preds, labels)

def segment_criterion(preds, labels):
    return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
           IoULoss().forward(preds, labels)

def contour_criterion(preds, labels):
    return IoULoss().forward(preds, labels)

def weight_criterion(preds, labels, weights):
    return binary_cross_entropy(preds, labels, weights) + \
           IoULoss().forward(preds, labels)

def focal_criterion(preds, labels, weights):
    return FocalLoss().forward(preds, labels, weights) + \
           IoULoss().forward(preds, labels)

def CORAL_loss(feature_map_s, feature_map_t):
    return CORAL().forward(feature_map_s, feature_map_t)

def MMD_loss(feature_map_s, feature_map_t):
    return MMD().forward(feature_map_s, feature_map_t)

def regularizer(feature_map_s, feature_map_t, loss):
    if loss == 'CORAL':
        return CORAL_loss(feature_map_s, feature_map_t)
    elif loss == 'MMD':
        return MMD_loss(feature_map_s, feature_map_t)
    else:
        raise NotImplementedError()
        return None