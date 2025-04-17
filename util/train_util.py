import copy
import logging
import math
import os
import pdb
import shutil
from collections import OrderedDict

import ignite.distributed as ignite_dist
import torch
from ignite.engine import Events
from ignite.handlers.param_scheduler import LRScheduler, PiecewiseLinear
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split

import util.yaml_utils as yaml_utils

logger = logging.getLogger()


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def set_learning_rate_scheduler(trainer, optimizer, optimizer_config, max_iter):
    if "lr_milestone" not in optimizer_config:
        return None
    milestone = optimizer_config["lr_milestone"]
    if milestone == "linear":
        init_lr = optimizer_config["args"]["lr"]
        lr_scheduler = PiecewiseLinear(
            optimizer, "lr", milestones_values=[(0, init_lr), (max_iter, 0)], save_history=True
        )
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    elif milestone == "cosine":
        warmup = optimizer_config["warmup"] if "warmup" in optimizer_config else 0
        lr_scheduler = LRScheduler(CosineAnnealingLR(optimizer, max_iter, num_warmup_steps=warmup))
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    else:
        gamma = optimizer_config["lr_drop_rate"]
        lr_scheduler = LRScheduler(MultiStepLR(optimizer, milestones=milestone, gamma=gamma), save_history=True)
        trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    return lr_scheduler


def CosineAnnealingLR(optimizer, max_iteration, num_warmup_steps=0, num_cycles=7.0 / 16.0, last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, max_iteration - num_warmup_steps))
        no_progress = min(1, max(no_progress, 0))
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))  # this is correct

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def create_result_dir(result_dir, config_path):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, "{}/{}".format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)


def load_models(model_config):
    model = yaml_utils.load_model(model_config["func"], model_config["name"], model_config["args"])
    if "pretrained" in model_config:
        state_dict = torch.load(model_config["pretrained"], map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


def desugar_keys(old_gen_dict):
    gen_dict = OrderedDict()
    for key, value in old_gen_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module.") :]
            print(f"Rename G's {key} ==> {key[len('module.'):]} for loading")
        gen_dict[new_key] = value
    return gen_dict


def load_loss_function(loss_config, suffix=None):
    loss_config_ = copy.deepcopy(loss_config)
    loss_name = loss_config_["name"]
    if suffix is not None:
        loss_config_["name"] = "_".join([loss_name, suffix])
    loss = yaml_utils.load_method(loss_config_)
    return loss


def make_optimizer(model, opt_config):
    # Select from https://pytorch.org/docs/stable/optim.html
    # NOTE: The order of the arguments for optimizers follows their definitions.
    opt = yaml_utils.load_optimizer(model, opt_config["algorithm"], args=opt_config["args"])
    return opt


def reduce_dataset(use_ratio, dataset):
    assert 0.0 < use_ratio and use_ratio <= 1.0
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    full_size = len(dataset)
    use_size = int(full_size * use_ratio)
    dataset, _ = random_split(dataset, [use_size, full_size - use_size])
    logger.info(f"## Reduced dataset size from {full_size} to {len(dataset)}")
    torch.manual_seed(seed)
    return dataset


def setup_train_dataloaders(config):
    # Dataset
    train = yaml_utils.load_dataset(config["dataset"])
    # DataLoader
    train_loader = DataLoader(train, config["batchsize"], num_workers=config["num_worker"], pin_memory=True)
    return train_loader


def setup_dataloaders(config, transform=None):
    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    all_train_dataset = yaml_utils.load_dataset(config["dataset"])
    if "use_ratio" in config["dataset"]:
        ratio = float(config["dataset"]["use_ratio"])
        all_train_dataset = reduce_dataset(ratio, all_train_dataset)
    train_size = int(len(all_train_dataset) * config["train_val_split_ratio"])
    val_size = len(all_train_dataset) - train_size
    train, val = random_split(all_train_dataset, [train_size, val_size])
    test = yaml_utils.load_dataset(config["dataset"], test=True)
    val.transform = test.transform

    # DataLoader
    train_loader = DataLoader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(val, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True)
    test_loader = DataLoader(
        test, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True
    )
    torch.manual_seed(seed)
    return train_loader, val_loader, test_loader


def setup_pretraining_dataloaders(config, transform=None):
    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    all_train_dataset = yaml_utils.load_dataset(config["dataset"])
    if "use_ratio" in config["dataset"]:
        ratio = float(config["dataset"]["use_ratio"])
        all_train_dataset = reduce_dataset(ratio, all_train_dataset)
    train_size = int(len(all_train_dataset) * config["train_val_split_ratio"])
    val_size = len(all_train_dataset) - train_size
    train, val = random_split(all_train_dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(val, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True)
    torch.manual_seed(seed)
    return train_loader, val_loader


def setup_test_dataloader(config, transform=None):
    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    dataset = yaml_utils.load_dataset(config["dataset_cls"], test=True)
    if "use_ratio" in config["dataset_cls"]:
        ratio = float(config["dataset_cls"]["use_ratio"])
        dataset = reduce_dataset(ratio, dataset)
    # DataLoader
    test_loader = DataLoader(
        dataset, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True
    )
    torch.manual_seed(seed)
    return test_loader
