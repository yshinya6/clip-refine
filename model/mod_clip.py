import pdb
from copy import deepcopy
from functools import partial

import clip
import clip.model
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from loralib.utils import apply_lora, mark_only_lora_as_trainable

OPENCLIP_DATASET = {"ViT-H-14": "laion2b_s32b_b79k", "ViT-bigG-14": "laion2b_s39b_b160k"}


def load_model(model_name: str):
    if model_name.startswith("Open_"):
        arch_name = model_name.split("_")[-1]
        model, _, preprocess = open_clip.create_model_and_transforms(arch_name, OPENCLIP_DATASET[arch_name])
        tokenizer = open_clip.get_tokenizer(arch_name)
    elif model_name == "siglip":
        model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
    elif model_name == "dfn":
        model, preprocess = open_clip.create_model_from_pretrained("ViT-H-14-378-quickgelu", pretrained="dfn5b")
        tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    else:  # Load CLIP models
        model, preprocess = clip.load(model_name)
        tokenizer = partial(clip.tokenize, truncate=True)
    return model, preprocess, tokenizer


class CLIP(torch.nn.Module):
    def __init__(
        self,
        backbone_name="ViT-B/32",
        feat_dim=2048,
        init_logit_scale=np.log(1 / 0.01),
        lora_params=None,
    ):
        super().__init__()
        # Load Backbone Vision-Language Model
        self.backbone, self.img_preprocess, self.tokenizer = load_model(backbone_name)
        if lora_params:
            lora_params["backbone"] = backbone_name
            apply_lora(lora_params, self.backbone)
            mark_only_lora_as_trainable(self.backbone)
        self.feat_dim = feat_dim
        self.convert_models_to_fp32()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * init_logit_scale)

    def convert_models_to_fp32(self):
        for p in self.parameters():
            p.data = p.data.float()

    def encode_image(self, x: torch.Tensor, normalized: bool = True):
        feat_v = self.backbone.encode_image(x)
        if normalized:
            feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)
        return feat_v

    def encode_text(self, t: torch.Tensor, normalized: bool = True):
        feat_t = self.backbone.encode_text(t)
        if normalized:
            feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
        return feat_t

    def forward(self, images: torch.Tensor, texts: torch.Tensor, test=False):
        feat_v = self.backbone.encode_image(images)
        feat_t = self.backbone.encode_text(texts)
        return {"image_features": feat_v, "text_features": feat_t, "logit_scale": self.logit_scale.exp()}


class ZeroshotClassifier(CLIP):
    def __init__(
        self,
        *args,
        pretrained_path,
        classname_file: str,
        class_prompt: str = "a photo of a",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        statedict = torch.load(pretrained_path, map_location="cpu")
        self.load_state_dict(statedict, strict=True)

        # Load class name dict for each label id
        self.id_classname_dict = self._make_classname_dict(classname_file)
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

    def get_text_features(self, device) -> torch.Tensor:
        if self._classname_features is not None:
            return self._classname_features

        class_texts = torch.cat([self.tokenizer(f"{self.class_prompt} {c}") for c in self.class_names]).to(device)
        text_features = self.encode_text(class_texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._classname_features = text_features
        return self._classname_features

    def forward(self, images, texts=None, test=False):
        # 1. Extract image features
        image_features = self.encode_image(images)

        # 2. Predict final labels
        class_similarities = (100.0 * image_features @ self.get_text_features(images.device).T).softmax(dim=-1)
        probs, preds = class_similarities.topk(1, dim=-1)
        modality_gap = F.mse_loss(image_features.mean(dim=0), self.get_text_features(images.device).mean(dim=0))
        output = {
            "preds": preds,
            "probs": probs,
            "feat_gap": modality_gap,
            "sparsity": 1.0,
        }
        return output
