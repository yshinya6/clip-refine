# dataloader here
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CONFIG = {
    "train_img_dir": "./dataset/coco/train2017",
    "train_annotation_file": "./dataset/coco/annotations/captions_train2017.json",
}


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_id_to_img_path[img_id] = file_name

    return img_id_to_img_path


def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations["annotations"]:
        img_id = caption_info["image_id"]
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []

        caption = caption_info["caption"]
        img_id_to_captions[img_id].append(caption)

    return img_id_to_captions


class COCOCaption(Dataset):

    def __init__(self, tokenizer, transform, test=False):

        super().__init__()

        self.config = CONFIG

        annotation_file = self.config["train_annotation_file"]
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())

        self.img_dir = self.config["train_img_dir"]

        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # randomly pick one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenizer(text)

        return img_input, text_input
