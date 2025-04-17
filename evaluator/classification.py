import pdb

import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batchsize = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.T
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul(100.0 / batchsize))
    return res


class ClassifierEvaluator:
    def __init__(
        self,
        *args,
        class_prompt: str = "a photo of",
        **kwargs,
    ):
        self.model = kwargs.pop("model")
        self.device = kwargs.pop("device")
        self.classname_file = kwargs.pop("classname_file")
        self.tokenizer = kwargs.pop("tokenizer")
        self.id_classname_dict = self._make_classname_dict(self.classname_file)
        self.class_names = list(self.id_classname_dict.values())
        self.class_prompt = class_prompt
        self._classname_features = None

    def _make_classname_dict(self, metadata_path: str):
        id_classname_dict = {}
        with open(metadata_path, "r") as f:
            for line in f:
                assert len(line.split("\t")) == 2, "metadata must be composed of lines of <class id>\\t<classname>"
                cls_id, cls_name = line.split("\t")
                id_classname_dict[cls_id] = cls_name.replace("\n", "").replace("_", " ")
        return id_classname_dict

    @torch.no_grad()
    def get_text_features(self, device) -> torch.Tensor:
        if self._classname_features is not None:
            return self._classname_features

        class_texts = torch.cat([self.tokenizer(f"{self.class_prompt} {c}") for c in self.class_names]).to(device)
        text_features = self.model.module.encode_text(class_texts, normalized=True)
        if isinstance(text_features, tuple):
            text_features = text_features[0]
        self._classname_features = text_features
        return self._classname_features

    def get_batch(self, batch, device=None):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=True),
            convert_tensor(y, device=device, non_blocking=True),
        )

    def __call__(self, engine, batch):
        self.model.eval()
        x, y = self.get_batch(batch, device=self.device)
        image_features = self.model.module.encode_image(x, normalized=True)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        y_pred = (100.0 * image_features @ self.get_text_features(x.device).T).softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y).detach().item()
        acc = accuracy(y_pred, y)[0].detach().item()
        return {"loss": loss, "accuracy": acc}


class FinetuningEvaluator:
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop("model")
        self.device = kwargs.pop("device")

    def get_batch(self, batch, device=None):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=True),
            convert_tensor(y, device=device, non_blocking=True),
        )

    def __call__(self, engine, batch):
        classifier = self.classifier
        classifier.eval()
        x, y = self.get_batch(batch, device=self.device)
        with torch.no_grad():
            y_pred = classifier(x, test=True)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        return (y_pred.detach(), y.detach())
