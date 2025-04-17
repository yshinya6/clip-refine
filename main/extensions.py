import json
import os
import pdb
import time


def register_metrics(engine, metrics):
    for key in metrics:
        metrics[key].attach(engine, key)


def check_accuracy(engine):
    score = engine.state.metrics["accuracy"]
    return score


def check_cls_loss(engine):
    score = engine.state.metrics["loss"]
    return -score


def check_loss(engine):
    score = engine.state.metrics["val_loss"]
    return -score


def log_training_results(engine, log, pbar):
    metrics = engine.state.metrics
    log_dict = {}
    lr = engine._process_function.opt.param_groups[0]["lr"]
    log_dict["lr"] = lr
    log_dict["epoch"] = engine.state.epoch
    log_dict["iteration"] = engine.state.iteration
    for m in metrics.keys():
        if m in ["y_pred", "y"]:
            continue
        log_dict[m] = metrics[m]
        msg += f"{m}: {metrics[m]:4f} "
    pbar.log_message(msg)
    log["running"].append(log_dict)


def log_training_results_mps(engine, log, pbar):
    metrics = engine.state.metrics
    log_dict = {}
    lr = engine._process_function.optimizer.param_groups[0]["lr"]
    log_dict["lr"] = lr
    log_dict["epoch"] = engine.state.epoch
    log_dict["elapsed_time"] = engine.state.times["EPOCH_COMPLETED"]
    msg = f"Training Results - Epoch: {engine.state.epoch} Iteration: {engine.state.iteration} time: {log_dict['elapsed_time']} lr: {lr:.6f} "
    for m in metrics.keys():
        if m in ["y_pred", "y"]:
            continue
        log_dict[m] = metrics[m]
        msg += f"{m}: {metrics[m]:4f} "
    pbar.log_message(msg)
    log["running"].append(log_dict)


def log_validation_results(engine, evaluator, val_loader, test_loader, log, pbar, dist):
    # Do validation
    evaluator.run(val_loader)
    log_dict = log["running"][-1]
    metrics = evaluator.state.metrics
    log_dict["val_accuracy"] = metrics["accuracy"]
    log_dict["val_loss"] = metrics["loss"]
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}".format(
            engine.state.epoch, metrics["accuracy"], metrics["loss"]
        )
    )
    # Do test iff the best validation accuracy was updated
    # if log["best_val_accuracy"] < log_dict["val_accuracy"] or log_dict["val_loss"] < log["best_val_loss"]:
    if log_dict["val_loss"] < log["best_val_loss"]:
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        log["test_accuracy"] = metrics["accuracy"]
    log["best_val_accuracy"] = max(log["best_val_accuracy"], log_dict["val_accuracy"])
    log["best_val_loss"] = min(log["best_val_loss"], log_dict["val_loss"])
    dump_log(log, dist)


def log_pretraining_validation_results(engine, cont_evaluator, val_loader, cls_evaluator, test_loader, log, pbar, dist):
    # Do validation on cotrastive learning
    cont_evaluator.run(val_loader)
    log_dict = log["running"][-1]
    metrics_cont = cont_evaluator.state.metrics
    log_dict["val_loss"] = metrics_cont["val_loss"]
    log_dict["val_feat_gap"] = metrics_cont["val_gap"]
    log_dict["val_alignment"] = metrics_cont["val_alignment"]
    log_dict["val_uniformity"] = metrics_cont["val_uniformity"]

    # Do validation on classification
    cls_evaluator.run(test_loader)
    metrics_cls = cls_evaluator.state.metrics
    log_dict["cls_accuracy"] = metrics_cls["accuracy"]
    log_dict["cls_loss"] = metrics_cls["loss"]

    pbar.log_message(
        "Validation Results - Epoch: {}  Avg loss: {:.4f} Avg feat gap: {:.4f} Avg aligment {:.4f} Avg uniformity {:.4f} Avg cls accuracy {:.4f} Avg cls loss {:.4f}".format(
            engine.state.epoch,
            metrics_cont["val_loss"],
            metrics_cont["val_gap"],
            metrics_cont["val_alignment"],
            metrics_cont["val_uniformity"],
            metrics_cls["accuracy"],
            metrics_cls["loss"],
        )
    )
    log["best_val_loss"] = min(log["best_val_loss"], log_dict["val_loss"])

    dump_log(log, dist)


def log_pretraining_results(engine, cls_evaluator, test_loader, log, pbar, dist):
    log_dict = log["running"][-1]

    # Do validation on classification
    cls_evaluator.run(test_loader)
    metrics_cls = cls_evaluator.state.metrics
    log_dict["cls_accuracy"] = metrics_cls["accuracy"]
    log_dict["cls_loss"] = metrics_cls["loss"]

    pbar.log_message(
        "Validation Results - Epoch: {} Iteration: {} Avg cls accuracy {:.4f} Avg cls loss {:.4f}".format(
            engine.state.epoch,
            engine.state.iteration,
            metrics_cls["accuracy"],
            metrics_cls["loss"],
        )
    )
    log["best_val_loss"] = min(log["best_val_loss"], log_dict["cls_loss"])

    dump_log(log, dist)


def dump_log(log, dist):
    with open(os.path.join(dist, "log"), "w") as f:
        json.dump(log, f, indent=2, sort_keys=True, separators=(",", ": "))


def load_log(path):
    with open(os.path.join(path, "log"), "r") as f:
        log = json.load(f)
    return log


def resume_training(engine, resume_epoch):
    engine.state.iteration = resume_epoch * len(engine.state.dataloader)
    engine.state.epoch = resume_epoch
