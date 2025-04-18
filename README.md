# Post-pre-training for Modality Alignment in Vision-Language Foundation Models (CVPR2025)
## Requirements
### Software Requirements
* CUDA >= 12.3
### Python Requirements
* Please see `apptainer/config.def`

## Preparations
### Post-pre-training Dataset: COCO Caption (2017)
  1. Download the dataset from [here](https://cocodataset.org/#home)
  2. Install the dataset into `./dataset/coco/`
### Evaluation Dataset: ImageNet
  1. Download the dataset from [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
  2. Install the dataset into `./dataset/imagenet/`

## Example
### Run Post-pre-training of CLIP-Refine on COCO Caption

```sh
python3 main/train.py --config_path config/01_post-pre-training/clip-refine.yaml
```

### Evaluate Zero-shot Performance on ImageNet

```sh
python3 main/test.py --config_path config/01_post-pre-training/clip-refine.yaml
```

## Citation
```bibtex
@inproceedings{Yamaguchi_CVPR25_CLIP-Refine,
  title={Post-pre-training for Modality Alignment in Vision-Language Foundation Models},
  author={Yamaguchi, Shin'ya and Feng, Dewei and Kanai, Sekitoshi and Adachi, Kazuki and Chijiwa, Daiki},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
