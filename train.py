import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import shutil

from dataset.dataset_loader import DatasetLoader
from dataset.sampler import TaskSampler

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.backbones.swiftformer import SwiftFormerBackbone
from models.cpea import CPEA
from models.model import SwiftCPEA

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig
import wandb


def _prepare_training_data(cfg):
    DATA_ROOT = Path(get_original_cwd()) / cfg.data.root
    train_set = DatasetLoader(root=str(DATA_ROOT), split='train',
                              max_imgs_per_class=cfg.data.max_imgs_per_class,
                              image_size=cfg.model.image_size)
    val_set   = DatasetLoader(root=str(DATA_ROOT), split='val',
                              max_imgs_per_class=cfg.data.max_imgs_per_class,
                              image_size=cfg.model.image_size)

    train_sampler = TaskSampler(
        train_set,
        n_way=cfg.data.n_way,
        n_shot=cfg.data.n_shot,
        n_query=cfg.data.n_query,
        n_tasks=cfg.data.n_tasks_train,
    )
    val_sampler = TaskSampler(
        val_set,
        n_way=cfg.data.n_way,
        n_shot=cfg.data.n_shot,
        n_query=cfg.data.n_query,
        n_tasks=cfg.data.n_tasks_val,
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=val_sampler.episodic_collate_fn,
    )
    return train_loader, val_loader

def _training_epoch(
    model,
    data_loader,
    optimizer_backbone,
    optimizer_head,
    loss_fn,
    device,
    freeze_backbone=False,
):
    all_loss = []
    correct, total = 0, 0
    model.train()
    if freeze_backbone:
        model.backbone.eval()

    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_idx, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

            if not freeze_backbone:
                optimizer_backbone.zero_grad()
            optimizer_head.zero_grad()
            logits = model(support_images, query_images, support_labels)
            # print(f"logits: {logits.shape}, query_labels: {query_labels.shape}")
            loss = loss_fn(logits, query_labels)
            loss.backward()
            if not freeze_backbone:
                optimizer_backbone.step()
            optimizer_head.step()

            preds = logits.argmax(dim=1)
            correct += (preds == query_labels).sum().item()
            total   += query_labels.size(0)

            all_loss.append(loss.item())
            tqdm_train.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss  = sum(all_loss) / len(all_loss)
    train_acc = correct / total
    return avg_loss, train_acc

def _set_backbone_trainable(model, trainable):
    for p in model.backbone.parameters():
        p.requires_grad_(trainable)

def _evaluate(model, data_loader, loss_fn, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels, _ in data_loader:
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

            logits = model(support_images, query_images, support_labels)
            loss = loss_fn(logits, query_labels)

            pred = torch.argmax(logits, dim=1)
            correct += (pred == query_labels).sum().item()
            total += query_labels.size(0)
            total_loss += loss.item()

        acc = correct / total
        avg_loss = total_loss / len(data_loader)

    return avg_loss, acc

def _save_metrics(history, log_dir):
    np.save(log_dir / "history.npy", history)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Evolution")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Evolution")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(log_dir / "metrics.png")
    plt.close()

@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig) -> None:
    # Fix for Windows symlink issues with Hugging Face / timm
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    
    # cfg = OmegaConf.load("configs/train.yaml")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")

    # Build run_name first so it can be passed directly to wandb.init
    run_name = f"{cfg.model.backbone_name}_{cfg.data.n_way}w{cfg.data.n_shot}s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="swiftcpea-classification",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="train"
    )

    # Prepare dataset
    train_loader, val_loader = _prepare_training_data(cfg=cfg)
    
    # Define models
    print(f"Using backbone: {cfg.model.backbone_name}")
    # Force absolute path for cache to avoid Hydra confusion
    abs_cache_dir = os.path.abspath(os.path.join(get_original_cwd(), cfg.paths.checkpoint_dir))
    
    backbone = SwiftFormerBackbone(
        name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
        cache_dir=abs_cache_dir
    )

    # Calculate seq_len
    dummy_input = torch.zeros(1, 3, cfg.model.image_size, cfg.model.image_size)
    feat_map = backbone.backbone.forward_features(dummy_input)
    seq_len = feat_map.shape[-2] * feat_map.shape[-1] # seq_len for CPEA: H*W
    
    # CPEA and model wrapper
    cpea = CPEA(
        in_dim=backbone.embed_dim,
        seq_len=seq_len,
        dropout=cfg.model.get("dropout", 0.1),
        class_aware_factor=cfg.model.get("class_aware_factor", 2.0),
    )
    model = SwiftCPEA(
        backbone=backbone,
        head=cpea,
        feat_dropout=cfg.model.get("feat_dropout", 0.0),
    ).to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    # Optimizers
    opt_name = cfg.training.get("optimizer", "SGD")
    OptimizerClass = getattr(torch.optim, opt_name)
    opt_kwargs = {"weight_decay": cfg.training.weight_decay}
    if opt_name == "SGD":
        opt_kwargs["momentum"] = cfg.training.get("momentum", 0.9)

    optimizer_backbone = OptimizerClass(
        model.backbone.parameters(),
        lr=cfg.training.lr_backbone,
        **opt_kwargs
    )
    optimizer_cpea = OptimizerClass(
        model.head.parameters(),
        lr=cfg.training.lr_head,
        **opt_kwargs
    )
    
    # Schedulers
    sched_name = cfg.training.get("scheduler", "CosineAnnealingLR")
    SchedulerClass = getattr(lr_scheduler, sched_name)

    if sched_name == "CosineAnnealingLR":
        sched_kwargs = {
            "T_max": cfg.training.n_epochs,
            "eta_min": cfg.training.get("scheduler_eta_min", 1e-7),
        }
    elif sched_name == "MultiStepLR":
        sched_kwargs = {
            "milestones": cfg.training.scheduler_milestones,
            "gamma": cfg.training.scheduler_gamma,
        }
    elif sched_name == "StepLR":
        sched_kwargs = {
            "step_size": cfg.training.get("scheduler_step_size", 10),
            "gamma": cfg.training.scheduler_gamma,
        }
    else:
        sched_kwargs = {}

    n_epochs = cfg.training.n_epochs
    freeze_epochs = cfg.training.get("warmup_freeze_epochs", 0)

    # When the backbone is frozen for the first N epochs, CosineAnnealingLR should
    # only count the *active* (unfrozen) epochs — otherwise the LR decays during
    # the warmup and wastes the cosine budget.
    active_epochs = n_epochs - freeze_epochs
    if sched_name == "CosineAnnealingLR":
        sched_kwargs["T_max"] = max(active_epochs, 1)

    scheduler_backbone = SchedulerClass(optimizer_backbone, **sched_kwargs)
    scheduler_cpea = SchedulerClass(optimizer_cpea, **sched_kwargs)

    # Run directory organization (run_name already defined above)
    run_dir = Path(get_original_cwd()) / "runs" / run_name
    
    checkpoint_dir = run_dir
    logs_dir = run_dir / "logs"
    config_save_dir = run_dir
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs
    # shutil.copytree(Path(get_original_cwd()) / "configs", config_save_dir, dirs_exist_ok=True)
    with open(run_dir / "configs.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    
    print(f"Run directory: {run_dir}")

    # Tensorboard
    tb_writer = SummaryWriter(logs_dir)

    best_val_acc = 0.0
    backbone_frozen = False  # track current state

    # Early stopping state
    es_patience = cfg.training.get("early_stopping_patience", 15)
    es_min_delta = cfg.training.get("early_stopping_min_delta", 0.001)
    es_counter = 0  # epochs without sufficient improvement

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    epoch_bar = tqdm(range(n_epochs), desc="Epochs", unit="ep")
    for epoch in epoch_bar:
        # ── Backbone freeze / unfreeze ─────────────────────────────────────
        if freeze_epochs > 0:
            if epoch < freeze_epochs and not backbone_frozen:
                _set_backbone_trainable(model, False)
                backbone_frozen = True
                tqdm.write(f"🔒 Backbone frozen for first {freeze_epochs} epochs")
            elif epoch >= freeze_epochs and backbone_frozen:
                _set_backbone_trainable(model, True)
                backbone_frozen = False
                tqdm.write(f"🔓 Backbone unfrozen at epoch {epoch + 1}")
                wandb.log({"backbone_unfrozen_epoch": epoch + 1})
        # ──────────────────────────────────────────────────────────────────

        train_loss, train_acc = _training_epoch(
            model,
            train_loader,
            optimizer_backbone,
            optimizer_cpea,
            loss_fn,
            device,
            freeze_backbone=backbone_frozen,
        )
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device)
        if not backbone_frozen:  # only tick backbone LR once it's actively training
            scheduler_backbone.step()
        scheduler_cpea.step()

        tqdm.write(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("train/acc", train_acc, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/acc", val_acc, epoch)

        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr/backbone": scheduler_backbone.get_last_lr()[0],
            "lr/head": scheduler_cpea.get_last_lr()[0],
            "epoch": epoch + 1,
        })

        # Update history and save plot
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        _save_metrics(history, checkpoint_dir)

        if val_acc > best_val_acc + es_min_delta:
            best_val_acc = val_acc
            es_counter   = 0  # reset patience counter
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_backbone_state_dict": optimizer_backbone.state_dict(),
                    "scheduler_backbone_state_dict": scheduler_backbone.state_dict(),
                    "optimizer_cpea_state_dict": optimizer_cpea.state_dict(),
                    "scheduler_cpea_state_dict": scheduler_cpea.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                checkpoint_dir / cfg.paths.checkpoint_name,
            )
            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1
            artifact = wandb.Artifact(name="best_model", type="model")
            artifact.add_file(str(checkpoint_dir / cfg.paths.checkpoint_name))
            wandb.log_artifact(artifact)
            tqdm.write(f"   ✓ New best saved — val_acc={val_acc:.4f} (epoch {epoch+1})")
        else:
            es_counter += 1
            tqdm.write(f"   Early stopping: no improvement for {es_counter}/{es_patience} epochs")
            if es_counter >= es_patience:
                tqdm.write(f"\n⛔ Early stopping triggered at epoch {epoch+1} — best val_acc={best_val_acc:.4f}")
                wandb.run.summary["stopped_epoch"] = epoch + 1
                wandb.run.summary["early_stopped"] = True
                break

    tb_writer.close()
    wandb.finish()

if __name__ == "__main__":
    train()
