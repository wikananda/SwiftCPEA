"""
test.py — Few-shot evaluation script.

Usage examples
--------------
# Load everything from a run folder (config + checkpoint):
    python test.py --run swiftformer_l3.dist_in1k_5w5s_20250420_143201

# Load from a run folder but override the checkpoint name:
    python test.py --run swiftformer_l3.dist_in1k_5w5s_20250420_143201 \
                   --checkpoint_name best_model.pth

# Force the static configs/test.yaml (old behaviour):
    python test.py --config

# Force the static config but point at a specific checkpoint:
    python test.py --config --checkpoint_dir cp --checkpoint_name swiftcpea.pth

# Override n_episodes regardless of config source:
    python test.py --run swiftformer_l3.dist_in1k_5w5s_20250420_143201 --n_episodes 1000
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from dataset.dataset_loader import DatasetLoader
from dataset.sampler import TaskSampler
from models.backbones.swiftformer import SwiftFormerBackbone
from models.cpea import CPEA
from models.model import SwiftCPEA


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_confidence_interval(accuracies: np.ndarray):
    """Returns (mean, 95% confidence interval half-width)."""
    mean = accuracies.mean()
    ci95 = 1.96 * accuracies.std() / np.sqrt(len(accuracies))
    return mean, ci95


def _load_config(args) -> OmegaConf:
    """
    Resolve the config for evaluation.

    --run <name>
        Loads runs/<name>/configs.yaml verbatim (same model/data/paths as training).
        Injects two test-only fields:
          • test.n_episodes   (default: 600)
          • paths.checkpoint_dir → redirected to the run folder
        All model/data settings (image_size, dropout, n_way, ...) come from the
        saved training config — guaranteed to match the checkpoint.

    --config
        Loads configs/test.yaml and resolves its Hydra `defaults` sub-configs.
        Use this if you need to manually override model or data settings.

    Both modes accept CLI overrides: --n_episodes, --checkpoint_name, --checkpoint_dir.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent
    N_EPISODES_DEFAULT = 600

    if args.run:
        run_dir = PROJECT_ROOT / "runs" / args.run
        cfg_path = run_dir / "configs.yaml"
        if not cfg_path.exists():
            available = [p.name for p in (PROJECT_ROOT / "runs").iterdir() if p.is_dir()]
            raise FileNotFoundError(
                f"No configs.yaml found in run folder: {run_dir}\n"
                f"Available runs: {available}"
            )

        # Load the training config exactly as saved — model/data params are identical to training
        cfg = OmegaConf.load(cfg_path)

        # Inject only the two test-specific fields the training config doesn't have
        cfg = OmegaConf.merge(cfg, {
            "test": {"n_episodes": N_EPISODES_DEFAULT},
            "paths": {
                # Redirect checkpoint_dir to the run folder (overrides the generic "cp" saved in config)
                "checkpoint_dir":  str(run_dir),
                "checkpoint_name": cfg.paths.get("checkpoint_name", "swiftcpea.pth"),
            },
        })
        print(f"📂 Loaded config from run: {run_dir}")

    else:
        # Escape hatch: load configs/test.yaml and resolve its Hydra `defaults` sub-configs.
        cfg_path = PROJECT_ROOT / "configs" / "test.yaml"
        raw = OmegaConf.load(cfg_path)
        print(f"📄 Loaded config from: {cfg_path}")

        # OmegaConf ListConfig entries are DictConfig, not plain dict — convert first.
        merged = OmegaConf.create({})
        for entry in OmegaConf.to_container(raw.get("defaults", []), resolve=False):
            if isinstance(entry, dict):
                for group, name in entry.items():
                    if group == "_self_":
                        continue
                    sub_path = PROJECT_ROOT / "configs" / group / f"{name}.yaml"
                    if sub_path.exists():
                        merged = OmegaConf.merge(merged, {group: OmegaConf.load(sub_path)})

        # Merge test.yaml's own keys (paths, test) on top
        cfg = OmegaConf.merge(merged, OmegaConf.masked_copy(raw, [k for k in raw if k != "defaults"]))

    # ── Apply CLI overrides (always highest priority) ──────────────────────────
    if args.n_episodes is not None:
        cfg = OmegaConf.merge(cfg, {"test": {"n_episodes": args.n_episodes}})
    if args.checkpoint_dir:
        cfg = OmegaConf.merge(cfg, {"paths": {"checkpoint_dir": args.checkpoint_dir}})
    if args.checkpoint_name:
        cfg = OmegaConf.merge(cfg, {"paths": {"checkpoint_name": args.checkpoint_name}})

    return cfg, PROJECT_ROOT



# ── Main ───────────────────────────────────────────────────────────────────────

def test(cfg, project_root: Path, run_name: str = None) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    wandb_name = f"test_{run_name}" if run_name else None
    wandb.init(
        project="swiftcpea-classification",
        name=wandb_name,
        job_type="eval",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Build model ────────────────────────────────────────────────────────────
    backbone = SwiftFormerBackbone(
        name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
    )

    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, cfg.model.image_size, cfg.model.image_size)
        feat_map = backbone.backbone.forward_features(dummy_input)
        seq_len = feat_map.shape[-2] * feat_map.shape[-1]

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

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_dir = cfg.paths.checkpoint_dir
    ckpt_name = cfg.paths.checkpoint_name

    # checkpoint_dir may be an absolute path (run folder) or relative to project root
    ckpt_dir_path = Path(ckpt_dir)
    if not ckpt_dir_path.is_absolute():
        ckpt_dir_path = project_root / ckpt_dir_path

    ckpt_path = ckpt_dir_path / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"🔍 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Dataset ────────────────────────────────────────────────────────────────
    DATA_ROOT = project_root / cfg.data.root
    test_set = DatasetLoader(
        root=str(DATA_ROOT),
        split="test",
        max_imgs_per_class=cfg.data.max_imgs_per_class,
        image_size=cfg.model.image_size,
    )
    test_sampler = TaskSampler(
        test_set,
        n_way=cfg.data.n_way,
        n_shot=cfg.data.n_shot,
        n_query=cfg.data.n_query,
        n_tasks=cfg.test.n_episodes,
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    model.eval()
    episode_accs = []
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels, _ in tqdm(test_loader, desc="Testing"):
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            logits = model(support_images, query_images, support_labels)
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item()
            episode_accs.append(acc)

    accs = np.array(episode_accs)
    mean, ci95 = compute_confidence_interval(accs)

    wandb.log({
        "test/accuracy_mean": mean * 100,
        "test/accuracy_ci95": ci95 * 100,
        "test/n_episodes":    len(accs),
    })
    wandb.run.summary.update({
        "test/accuracy_mean": mean * 100,
        "test/accuracy_ci95": ci95 * 100,
    })
    wandb.finish()

    print(f"\nTest Results over {len(accs)} episodes:")
    print(f"    Accuracy: {mean*100:.2f}% ± {ci95*100:.2f}%")


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a few-shot model. Pass --run to load from a run folder, "
                    "or --config to use configs/test.yaml."
    )

    # Config source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run", metavar="RUN_NAME",
        help="Name of the run folder under runs/. Loads runs/<RUN_NAME>/configs.yaml "
             "and runs/<RUN_NAME>/swiftcpea.pth by default.",
    )
    source.add_argument(
        "--config", action="store_true",
        help="Use configs/test.yaml (static fallback config).",
    )

    # Optional overrides
    parser.add_argument("--checkpoint_dir",  default=None,
                        help="Override checkpoint directory (absolute or relative to project root).")
    parser.add_argument("--checkpoint_name", default=None,
                        help="Override checkpoint filename (default: swiftcpea.pth).")
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Override number of test episodes.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg, root = _load_config(args)
    test(cfg, root, run_name=args.run)