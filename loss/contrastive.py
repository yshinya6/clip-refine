import torch
import torch.nn.functional as F


def clip_loss(image_features, text_features, logit_scale):
    device = image_features.device
    logits_per_image = image_features @ text_features.T
    logits_per_text = text_features @ image_features.T
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss = (
        F.cross_entropy(logit_scale * logits_per_image, labels) + F.cross_entropy(logit_scale * logits_per_text, labels)
    ) / 2

    return total_loss
