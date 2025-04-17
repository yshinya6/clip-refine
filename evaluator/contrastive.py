import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor

from loss.contrastive import clip_loss


def align(x: torch.Tensor, y: torch.Tensor, alpha=2):
    pos_align = (x - y).norm(dim=1).pow(alpha).mean()
    return pos_align


def uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class ContrativeEvaluator:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model")
        self.device = kwargs.pop("device")
        self.loss = clip_loss

    def get_batch(self, batch, device=None):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=True),
            convert_tensor(y, device=device, non_blocking=True),
        )

    def __call__(self, engine, batch):
        report = {}
        self.model.eval()
        images, texts = self.get_batch(batch, device=self.device)
        with torch.no_grad():
            out = self.model(images, texts.squeeze())
        feat_i, feat_t = out["image_features"], out["text_features"]
        contrastive_loss = self.loss(feat_i, feat_t, out["logit_scale"])
        feat_gap = F.pairwise_distance(feat_i, feat_t).mean()
        alignment_loss = align(feat_t, feat_i)
        uniformity_loss = uniformity(torch.cat([feat_i, feat_t], dim=0))
        report.update(
            {
                "loss": contrastive_loss.detach().item(),
                "feat_gap": feat_gap.detach().item(),
                "alignment": alignment_loss.detach().item(),
                "uniformity": uniformity_loss.detach().item(),
            }
        )
        return report
