"""
Script for testing TaskSampler + MiniImageNet
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader
from dataset.mini_imagenet import MiniImageNet
from dataset.sampler import TaskSampler

import torch

DATA_ROOT = ROOT / "data" / "mini_imagenet"
N_WAY = 5
N_SHOT = 5
N_QUERY = 15
N_TASKS = 3

# ---- 1. dataset ----
print("[1/4] Loading dataset...")
ds = MiniImageNet(root=str(DATA_ROOT), split="train")
print(f"{len(ds)} images, {len(ds.class_names)} classes")
assert len(ds.class_names) == 64, f"Expected 64 train classes, got {len(ds.class_names)}"

# ---- 2. sampler ----
print("[2/4] Building TaskSampler...")
sampler = TaskSampler(ds, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TASKS)
assert len(sampler) == N_TASKS
print(f"n_way={N_WAY}, n_shot={N_SHOT}, n_query={N_QUERY}, n_tasks={N_TASKS}")

# ---- 3. dataloader ----
print("[3/4] Building DataLoader with episodic_collate_fn...")
loader = DataLoader(
    ds,
    batch_sampler=sampler,
    num_workers=0,
    collate_fn=sampler.episodic_collate_fn,
)

# ---- 4. iterate episodes ----
print("[4/4] Running episodes...\n")
for episode_idx, (support_imgs, support_labels, query_imgs, query_labels, class_ids) in enumerate(loader):
    print(f"Episode {episode_idx + 1}:")
    # support query count
    for c in torch.unique(support_labels):
        support_count = (support_labels == c).sum()
        query_count = (query_labels == c).sum()
        print(f"Class {c}: support={support_count}, query={query_count}")
    print(f"support_imgs: {tuple(support_imgs.shape)} (expected: {N_WAY * N_SHOT}, 3, 84, 84)")
    print(f"support_labels: {tuple(support_labels.shape)} values={support_labels.tolist()}")
    print(f"query_imgs: {tuple(query_imgs.shape)} (expected: {N_WAY * N_QUERY}, 3, 84, 84)")
    print(f"query_labels: {tuple(query_labels.shape)} unique={sorted(set(query_labels.tolist()))}")
    print(f"class_ids: {class_ids}")

    # shape assertions
    assert support_imgs.shape == (N_WAY * N_SHOT,  3, 84, 84), "Wrong support image shape"
    assert query_imgs.shape == (N_WAY * N_QUERY, 3, 84, 84), "Wrong query image shape"
    assert support_labels.shape == (N_WAY * N_SHOT,), "Wrong support label shape"
    assert query_labels.shape == (N_WAY * N_QUERY,), "Wrong query label shape"

    # label range should be 0..N_WAY-1
    assert support_labels.min() >= 0 and support_labels.max() < N_WAY
    assert query_labels.min() >= 0 and query_labels.max() < N_WAY
    print(f"✅ All assertions passed\n")

print("✅ TaskSampler is working correctly!")
