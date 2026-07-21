import os
import tarfile
import urllib.request
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from src.config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, DEVICE
)

# MPS does not support pin_memory; disable to avoid warnings.
PIN_MEMORY = False if DEVICE.type == 'mps' else True

IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute"
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _download_imagenette():
    imagenette_dir = os.path.join(DATA_DIR, "imagenette2-320")
    if os.path.isdir(imagenette_dir):
        print("ImageNette already downloaded.")
        return imagenette_dir

    os.makedirs(DATA_DIR, exist_ok=True)
    url      = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tgz_path = os.path.join(DATA_DIR, "imagenette2-320.tgz")

    print(" Downloading ImageNette-320 (~1.5 GB) …")

    def _reporthook(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // max(total_size, 1))
        if count % 300 == 0:
            print(f"  {pct}%", end="\r", flush=True)

    urllib.request.urlretrieve(url, tgz_path, reporthook=_reporthook)
    print("\nExtracting …")

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)

    os.remove(tgz_path)
    print(" Done.")
    return imagenette_dir


def get_loaders(download: bool = True):
    imagenette_dir = _download_imagenette() if download else os.path.join(DATA_DIR, "imagenette2-320")

    train_root = os.path.join(imagenette_dir, "train")
    val_root   = os.path.join(imagenette_dir, "val")

    full_train = datasets.ImageFolder(train_root, transform=TRAIN_TRANSFORM)
    test_ds    = datasets.ImageFolder(val_root,   transform=EVAL_TRANSFORM)

    n       = len(full_train)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    split    = int(0.8 * n)
    train_ds = Subset(full_train, indices[:split])

    val_full = datasets.ImageFolder(train_root, transform=EVAL_TRANSFORM)
    val_ds   = Subset(val_full, indices[split:])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def get_test_subset(n: int = 200, seed: int = 0) -> tuple:
    """Legacy sampler: independently draws n images from the full val pool.

    WARNING: calls with different seeds are NOT guaranteed disjoint, since
    each call reseeds and samples from the same underlying pool of val
    images. Do not use this to build a tuning set and an evaluation set
    that need to be disjoint (e.g. feature-squeezing threshold tuning vs.
    the final 1000-image evaluation) -- use get_partition() instead, which
    guarantees no index overlap across named blocks.
    """
    imagenette_dir = os.path.join(DATA_DIR, "imagenette2-320")
    val_root = os.path.join(imagenette_dir, "val")
    test_ds  = datasets.ImageFolder(val_root, transform=EVAL_TRANSFORM)

    np.random.seed(seed)
    indices = np.random.choice(len(test_ds), size=min(n, len(test_ds)), replace=False)
    subset  = Subset(test_ds, indices)
    loader  = DataLoader(subset, batch_size=n, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)


# Disjoint partitioning for the evaluation pipeline

_PARTITION_SEED = 2024  # fixed seed for the one-time partition shuffle


_PARTITION_SPEC = [
    ("stage1_bench",  250),   # Stage 1 classical detector benchmark (Table I)
    ("fs_tuning",     500),   # Stage 2 feature-squeezing threshold tuning (theta)
    ("fs_dev",        200),   # Stage 2 standalone dev-set sanity check
    ("stage3_dev",    100),   # Stage 3 GradCAM qualitative dev set
    ("final_eval",   1000),   # Final 5-method comparison: Table II, Fig 8-13
]


def _full_val_size() -> int:
    imagenette_dir = os.path.join(DATA_DIR, "imagenette2-320")
    val_root = os.path.join(imagenette_dir, "val")
    test_ds  = datasets.ImageFolder(val_root, transform=EVAL_TRANSFORM)
    return len(test_ds)


def build_partition_indices(spec: list = None, seed: int = _PARTITION_SEED) -> dict:
    
    spec = spec if spec is not None else _PARTITION_SPEC
    pool_size = _full_val_size()
    total_needed = sum(n for _, n in spec)
    if total_needed > pool_size:
        raise ValueError(
            f"Partition spec requests {total_needed} images but val pool "
            f"only has {pool_size}. Reduce block sizes."
        )

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(pool_size)

    partitions = {}
    cursor = 0
    for name, n in spec:
        partitions[name] = shuffled[cursor:cursor + n]
        cursor += n

    seen = set()
    for name, idx in partitions.items():
        idx_set = set(idx.tolist())
        overlap = seen & idx_set
        if overlap:
            raise RuntimeError(
                f"Partition '{name}' overlaps with a previous block on "
                f"{len(overlap)} indices -- this should be impossible "
                f"with simple slicing and indicates a bug."
            )
        seen |= idx_set

    return partitions


_cached_partitions = None


def get_partition(block_name: str, partitions: dict = None) -> tuple:
    
    global _cached_partitions
    if partitions is None:
        if _cached_partitions is None:
            _cached_partitions = build_partition_indices()
        partitions = _cached_partitions

    if block_name not in partitions:
        raise KeyError(
            f"Unknown block '{block_name}'. Available: {list(partitions.keys())}"
        )

    imagenette_dir = os.path.join(DATA_DIR, "imagenette2-320")
    val_root = os.path.join(imagenette_dir, "val")
    test_ds  = datasets.ImageFolder(val_root, transform=EVAL_TRANSFORM)

    indices = partitions[block_name]
    subset  = Subset(test_ds, indices)
    loader  = DataLoader(subset, batch_size=len(indices), shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device)
    if tensor.dim() == 3:
        mean = mean[:, None, None]
        std  = std[:, None, None]
    else:
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    return (tensor * std + mean).clamp(0, 1)


def verify_clean_accuracy(model, loader, max_batches: int = 20) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= max_batches:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100 * correct / total
    print(f" Clean accuracy: {acc:.1f}%  ({correct}/{total})")
    return acc