import argparse
import glob
import json
import logging
import os
import pdb
import sys

import torch
import yaml
from torch.backends import cudnn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main.experimental_settings import SETTINGS
from util import yaml_utils
from util.train_util import create_result_dir, load_models, setup_test_dataloader

logging.basicConfig(level=logging.INFO)


def log_basic_info(logger, config):
    logger.info(f"- PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("--------------")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("--------------")


def find_best_model_path(directory):
    pattern = os.path.join(directory, "best_model*.pt")
    files = glob.glob(pattern)
    return files[0] if len(files) > 0 else None


def main(config):
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    logger = logging.getLogger(__name__)
    log_basic_info(logger, config)

    logger.info("Creating output directory")
    out = os.path.join(config["results_dir"], config["pattern"], f"experiment{config['experiment_id']}")
    create_result_dir(out, config["config_path"])

    logger.info("## Loading model")
    pretrained_path = find_best_model_path(out)
    if pretrained_path is not None:
        logger.info(f"### Found pre-trained weight: {pretrained_path}")
        config["models"]["model"]["pretrained"] = pretrained_path
    model = load_models(config["models"]["model"])
    model.to(device)
    model.eval()
    result_dict = {}

    for expr_name, config_target in SETTINGS["experiments"].items():
        logger.info(f"## Start {expr_name}")
        logger.info("### Loading target dataset")
        config_target["dataset_cls"]["args"]["transform"] = model.img_preprocess
        config_target["dataset_cls"]["args"]["tokenizer"] = model.tokenizer
        config_target["batchsize"] = config["eval_batchsize"]
        config_target["num_worker"] = config["num_worker"]
        test_loader = setup_test_dataloader(config_target)
        test_function = yaml_utils.load_module(config_target["test_function"], config_target["function_name"])
        args = {"model": model, "dataloader": test_loader, "device": device}
        if "test_args" in config_target:
            args.update(config_target["test_args"])
        logger.info("### Running test")
        metrics = test_function(**args)
        result_dict[expr_name] = metrics
        logger.info(f"## End of {expr_name}: {metrics}")

    # Report
    result_path = os.path.join(out, "test_result.json")
    with open(result_path, mode="w") as f:
        json.dump(result_dict, f, indent=2)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/base.yml", help="path to config file")
    parser.add_argument("--results_dir", type=str, default="./result", help="directory to save the results to")
    parser.add_argument("--resume", type=str, default="", help="path to the snapshot")
    parser.add_argument("--eval_batchsize", type=int, default=512, help="batchsize for evaluation")
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_id", type=int, default=0)

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    config.config_dict["pattern"] = yaml_utils.make_pattern(config)
    config.config_dict.update(vars(args))
    main(config.config_dict)


if __name__ == "__main__":
    run()
