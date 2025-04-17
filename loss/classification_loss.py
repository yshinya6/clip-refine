import torch


def naive_cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()
