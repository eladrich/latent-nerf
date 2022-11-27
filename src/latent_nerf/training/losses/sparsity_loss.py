import torch


def sparsity_loss(pred_ws):
    alphas = pred_ws.clamp(1e-5, 1 - 1e-5)

    loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
    return loss_entropy
