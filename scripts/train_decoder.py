import argparse
import bisect
import csv
import gc
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import UltrasoundFrames, file_digest
from vjepa2_decoder.models.vit_decoder import ViTDecoder


# ============================================================
# V-JEPA configurations
# ============================================================

VJEPA_MODELS = {
    "vjepa2": {
        "display_name": "V-JEPA 2 ViT-L",
        "hub_name": "vjepa2_vit_large",
        "checkpoint_url": (
            "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"
        ),
        "checkpoint_filename": "vitl.pt",
        "checkpoint_key": "target_encoder",
        "img_size": 256,
        "temporal_frames": 2,
        "strict": False,
    },

    "vjepa2_1": {
        "display_name": "V-JEPA 2.1 ViT-L",
        "hub_name": "vjepa2_1_vit_large_384",
        "checkpoint_url": (
            "https://dl.fbaipublicfiles.com/vjepa2/"
            "vjepa2_1_vitl_dist_vitG_384.pt"
        ),
        "checkpoint_filename": (
            "vjepa2_1_vitl_dist_vitG_384.pt"
        ),
        "checkpoint_key": "ema_encoder",
        "img_size": 384,
        "temporal_frames": 1,
        "strict": True,
    },
}


# ============================================================
# GPU helpers
# ============================================================

def clear_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_gpu_memory_stats():
    if not torch.cuda.is_available():
        return

    for gpu_index in range(
        torch.cuda.device_count()
    ):
        torch.cuda.reset_peak_memory_stats(
            gpu_index
        )


def print_gpu_memory(label):
    if not torch.cuda.is_available():
        return

    print()
    print(
        f"GPU memory: {label}",
        flush=True,
    )

    for gpu_index in range(
        torch.cuda.device_count()
    ):
        current_allocated = (
            torch.cuda.memory_allocated(
                gpu_index
            )
            / 1024**3
        )

        current_reserved = (
            torch.cuda.memory_reserved(
                gpu_index
            )
            / 1024**3
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated(
                gpu_index
            )
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved(
                gpu_index
            )
            / 1024**3
        )

        total_memory = (
            torch.cuda.get_device_properties(
                gpu_index
            ).total_memory
            / 1024**3
        )

        print(
            f"  GPU {gpu_index}: "
            f"current={current_allocated:.2f} GB | "
            f"reserved={current_reserved:.2f} GB | "
            f"peak={peak_allocated:.2f} GB | "
            f"peak reserved={peak_reserved:.2f} GB | "
            f"total={total_memory:.2f} GB",
            flush=True,
        )


# ============================================================
# General helpers
# ============================================================

def check_finite(
    tensor,
    name,
):
    if not torch.isfinite(
        tensor
    ).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf."
        )


def clean_backbone_state_dict(
    state_dict,
):
    cleaned = {}

    for key, value in state_dict.items():

        key = key.replace(
            "module.",
            "",
        )

        key = key.replace(
            "backbone.",
            "",
        )

        cleaned[key] = value

    return cleaned


def resize_frames(
    frames,
    img_size,
):
    if (
        frames.shape[-2] == img_size
        and frames.shape[-1] == img_size
    ):
        return frames

    return F.interpolate(
        frames,
        size=(
            img_size,
            img_size,
        ),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def make_video_input(
    frames,
    temporal_frames,
):
    video = frames.unsqueeze(
        2
    )

    if temporal_frames > 1:
        video = video.repeat(
            1,
            1,
            temporal_frames,
            1,
            1,
        )

    return video


def make_preview_indices(
    dataset_length,
    count,
):
    count = min(
        count,
        dataset_length,
    )

    if count <= 0:
        return []

    if count == 1:
        return [0]

    return [
        round(
            i
            * (
                dataset_length
                - 1
            )
            / (
                count
                - 1
            )
        )
        for i in range(
            count
        )
    ]


# ============================================================
# Checkpoints
# ============================================================

def checkpoint_path_for(
    model_key,
    checkpoint_dir,
):
    return (
        Path(
            checkpoint_dir
        )
        / VJEPA_MODELS[
            model_key
        ][
            "checkpoint_filename"
        ]
    )


def download_checkpoint(
    model_key,
    checkpoint_dir,
):
    config = VJEPA_MODELS[
        model_key
    ]

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        checkpoint_path_for(
            model_key,
            checkpoint_dir,
        )
    )

    print()
    print(
        f"Downloading/checking "
        f"{config['display_name']}"
    )

    print(
        f"Destination: "
        f"{destination}",
        flush=True,
    )

    torch.hub.load_state_dict_from_url(
        config[
            "checkpoint_url"
        ],
        model_dir=str(
            checkpoint_dir
        ),
        file_name=config[
            "checkpoint_filename"
        ],
        map_location="cpu",
        progress=True,
    )

    print(
        f"Checkpoint ready: "
        f"{destination}",
        flush=True,
    )


def load_local_checkpoint(
    model_key,
    checkpoint_dir,
):
    path = checkpoint_path_for(
        model_key,
        checkpoint_dir,
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing checkpoint: {path}\n\n"
            f"Run:\n"
            f"python scripts/train_decoder.py "
            f"--model {model_key} "
            f"--download-only"
        )

    print(
        f"Loading checkpoint: "
        f"{path}",
        flush=True,
    )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


# ============================================================
# Encoder
# ============================================================

def load_encoder(
    model_key,
    args,
    device,
):
    config = VJEPA_MODELS[
        model_key
    ]

    repo = Path(
        args.vjepa_repo
    ).resolve()

    if not (
        repo
        / "hubconf.py"
    ).exists():
        raise RuntimeError(
            f"No hubconf.py in {repo}"
        )

    print()
    print("=" * 70)

    print(
        f"Loading "
        f"{config['display_name']}"
    )

    print(
        f"Resolution: "
        f"{config['img_size']}x"
        f"{config['img_size']}"
    )

    print(
        f"Temporal frames: "
        f"{config['temporal_frames']}"
    )

    print("=" * 70)

    encoder, predictor = (
        torch.hub.load(
            str(repo),
            config[
                "hub_name"
            ],
            source="local",
            pretrained=False,
        )
    )

    del predictor

    checkpoint = (
        load_local_checkpoint(
            model_key,
            args.checkpoint_dir,
        )
    )

    key = config[
        "checkpoint_key"
    ]

    if key not in checkpoint:
        raise RuntimeError(
            f"Checkpoint missing key: "
            f"{key}"
        )

    state = (
        clean_backbone_state_dict(
            checkpoint[
                key
            ]
        )
    )

    result = (
        encoder.load_state_dict(
            state,
            strict=config[
                "strict"
            ],
        )
    )

    print(
        f"Encoder load result: "
        f"{result}",
        flush=True,
    )

    del checkpoint
    del state

    encoder = (
        encoder
        .eval()
        .requires_grad_(False)
        .to(device)
    )

    return encoder


def encode_batch(
    encoder,
    frames,
    config,
    device,
    bf16_supported,
):
    frames = resize_frames(
        frames,
        config[
            "img_size"
        ],
    )

    frames = frames.to(
        device,
        non_blocking=True,
    )

    video = make_video_input(
        frames,
        config[
            "temporal_frames"
        ],
    )

    with torch.no_grad():

        if bf16_supported:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                features = encoder(
                    video
                )

        else:
            features = encoder(
                video.float()
            )

    return (
        features,
        frames,
    )


# ============================================================
# Cache helpers
# ============================================================

def cache_manifest_path(
    directory,
):
    return (
        Path(directory)
        / "manifest.json"
    )


def read_manifest(
    directory,
):
    path = cache_manifest_path(
        directory
    )

    if not path.exists():
        return None

    with path.open(
        "r"
    ) as file:

        return json.load(
            file
        )


def cache_is_complete(
    directory,
    expected_samples,
    identity,
):
    directory = Path(
        directory
    )

    manifest = read_manifest(
        directory
    )

    if manifest is None:
        return False
    if manifest.get("identity") != identity:
        return False
    for name, digest in manifest.get("shard_digests", {}).items():
        path = directory / name
        if not path.is_file() or file_digest(path) != digest:
            return False
    if len(manifest.get("shard_digests", {})) != manifest.get("shards"):
        return False

    if (
        manifest.get(
            "samples"
        )
        != expected_samples
    ):
        return False

    shard_files = sorted(
        directory.glob(
            "shard_*.pt"
        )
    )

    return (
        len(shard_files)
        == manifest.get(
            "shards"
        )
    )


# ============================================================
# Create cache shards
# ============================================================

def cache_split(
    split_name,
    dataset,
    encoder,
    model_key,
    config,
    args,
    encoder_device,
    bf16_supported,
):
    cache_dir = (
        Path(
            args.cache_dir
        )
        / model_key
        / split_name
    )

    identity = dict(dataset=dataset.fingerprint, model=config,
                    encoder_sha256=file_digest(checkpoint_path_for(model_key, args.checkpoint_dir)),
                    cache_dtype=args.cache_dtype, target_dtype="float16",
                    mixing="global-random-training-shards-v1", cache_batch_size=args.cache_batch_size,
                    source_sha256=file_digest(Path(args.vjepa_repo) / "src/hub/backbones.py"))
    if (
        args.rebuild_cache
        and cache_dir.exists()
    ):
        shutil.rmtree(
            cache_dir
        )

    if cache_is_complete(
        cache_dir,
        len(dataset),
        identity,
    ):
        manifest = read_manifest(
            cache_dir
        )

        print()
        print(
            f"Using completed "
            f"{split_name} cache:"
        )

        print(
            f"  {cache_dir}"
        )

        print(
            f"  samples: "
            f"{manifest['samples']:,}"
        )

        print(
            f"  shards: "
            f"{manifest['shards']:,}",
            flush=True,
        )

        return cache_dir

    if cache_dir.exists():

        contents = list(
            cache_dir.iterdir()
        )

        if contents:

            print()
            print(
                f"Removing incomplete/old "
                f"cache:"
            )

            print(
                f"  {cache_dir}",
                flush=True,
            )

            shutil.rmtree(
                cache_dir
            )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=(
            args.cache_batch_size
        ),
        shuffle=(split_name == "train"),
        generator=torch.Generator().manual_seed(42),
        num_workers=(
            args.num_workers
        ),
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    print()
    print(
        f"Caching {split_name} "
        f"features for "
        f"{config['display_name']}"
    )

    print(
        f"Samples: "
        f"{len(dataset):,}"
    )

    print(
        f"Cache batch/shard size: "
        f"{args.cache_batch_size}"
    )

    print(
        f"Destination: "
        f"{cache_dir}",
        flush=True,
    )

    reset_gpu_memory_stats()

    sample_count = 0
    shard_count = 0
    first_batch = True

    for (
        batch_index,
        frames,
    ) in enumerate(
        loader
    ):

        features, targets = (
            encode_batch(
                encoder,
                frames,
                config,
                encoder_device,
                bf16_supported,
            )
        )

        check_finite(
            features,
            "Encoder features",
        )

        # ----------------------------------------------------
        # Make tensors compact and independent before saving.
        # ----------------------------------------------------

        features = (
            features
            .detach()
            .to("cpu")
            .contiguous()
            .clone()
        )

        targets = (
            targets
            .detach()
            .to("cpu")
            .contiguous()
            .clone()
        )

        if (
            args.cache_dtype
            == "bfloat16"
        ):
            features = features.to(
                torch.bfloat16
            )

        elif (
            args.cache_dtype
            == "float16"
        ):
            features = features.to(
                torch.float16
            )

        else:
            features = features.to(
                torch.float32
            )

        targets = targets.to(
            torch.float16
        )

        features = (
            features
            .contiguous()
            .clone()
        )

        targets = (
            targets
            .contiguous()
            .clone()
        )

        shard_path = (
            cache_dir
            / f"shard_{shard_count:05d}.pt"
        )

        temporary_path = (
            cache_dir
            / f"shard_{shard_count:05d}.tmp"
        )

        torch.save(
            {
                "features": features,
                "targets": targets,
            },
            temporary_path,
        )

        temporary_path.replace(
            shard_path
        )

        current_samples = (
            features.shape[0]
        )

        sample_count += (
            current_samples
        )

        shard_count += 1

        if first_batch:

            first_batch = False

            feature_bytes = (
                features.numel()
                * features.element_size()
            )

            target_bytes = (
                targets.numel()
                * targets.element_size()
            )

            shard_bytes = (
                feature_bytes
                + target_bytes
            )

            bytes_per_sample = (
                shard_bytes
                / current_samples
            )

            estimated_total = (
                bytes_per_sample
                * len(dataset)
            )

            print()
            print(
                "Cache size estimate:"
            )

            print(
                f"  feature shape: "
                f"{tuple(features.shape)}"
            )

            print(
                f"  target shape: "
                f"{tuple(targets.shape)}"
            )

            print(
                f"  feature dtype: "
                f"{features.dtype}"
            )

            print(
                f"  target dtype: "
                f"{targets.dtype}"
            )

            print(
                f"  shard size: "
                f"{shard_bytes / 1024**2:.2f} MB"
            )

            print(
                f"  total estimate: "
                f"{estimated_total / 1024**3:.2f} GB",
                flush=True,
            )

        if (
            shard_count == 1
            or sample_count % 512
            < current_samples
        ):

            print(
                f"{split_name}: "
                f"{sample_count:,}/"
                f"{len(dataset):,} "
                f"({shard_count:,} shards)",
                flush=True,
            )

    manifest = {
        "format_version": 3,
        "identity": identity,
        "shard_digests": {p.name: file_digest(p) for p in sorted(cache_dir.glob("shard_*.pt"))},
        "model": model_key,
        "split": split_name,
        "samples": sample_count,
        "shards": shard_count,
        "cache_batch_size": (
            args.cache_batch_size
        ),
        "feature_dtype": (
            args.cache_dtype
        ),
        "target_dtype": (
            "float16"
        ),
        "img_size": (
            config[
                "img_size"
            ]
        ),
    }

    with (
        cache_manifest_path(
            cache_dir
        )
        .open("w")
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        f"Finished {split_name}: "
        f"{sample_count:,} samples "
        f"in {shard_count:,} shards",
        flush=True,
    )

    print_gpu_memory(
        f"after caching "
        f"{model_key}/{split_name}"
    )

    return cache_dir


# ============================================================
# Sharded cache reader
# ============================================================

class ShardedFeatureCache:

    def __init__(
        self,
        directory,
    ):
        self.directory = Path(
            directory
        )

        manifest = read_manifest(
            self.directory
        )

        if manifest is None:
            raise RuntimeError(
                f"No valid cache manifest: "
                f"{self.directory}"
            )

        self.files = sorted(
            self.directory.glob(
                "shard_*.pt"
            )
        )

        if not self.files:
            raise RuntimeError(
                f"No cache shards found in "
                f"{self.directory}"
            )

        self.shard_sizes = []
        self.cumulative = []

        total = 0

        for file_path in self.files:

            shard = torch.load(
                file_path,
                map_location="cpu",
                weights_only=True,
            )

            count = (
                shard[
                    "features"
                ].shape[0]
            )

            self.shard_sizes.append(
                count
            )

            total += count

            self.cumulative.append(
                total
            )

            del shard

        self.total_samples = total

        expected = manifest[
            "samples"
        ]

        if (
            self.total_samples
            != expected
        ):
            raise RuntimeError(
                f"Cache sample mismatch: "
                f"{self.total_samples} "
                f"!= {expected}"
            )

    def __len__(
        self,
    ):
        return self.total_samples

    def load_shard(
        self,
        shard_index,
    ):
        return torch.load(
            self.files[
                shard_index
            ],
            map_location="cpu",
            weights_only=True,
        )

    def get_sample(
        self,
        global_index,
    ):
        if (
            global_index < 0
            or global_index
            >= self.total_samples
        ):
            raise IndexError(
                global_index
            )

        shard_index = (
            bisect.bisect_right(
                self.cumulative,
                global_index,
            )
        )

        previous = (
            0
            if shard_index == 0
            else self.cumulative[
                shard_index - 1
            ]
        )

        local_index = (
            global_index
            - previous
        )

        shard = self.load_shard(
            shard_index
        )

        features = (
            shard[
                "features"
            ][
                local_index
            ]
            .clone()
        )

        target = (
            shard[
                "targets"
            ][
                local_index
            ]
            .clone()
        )

        return (
            features,
            target,
        )


# ============================================================
# IMPORTANT FIX:
# decoder batches can span multiple shards
# ============================================================

def iter_cache_batches(
    cache,
    batch_size,
    shuffle,
    seed,
):
    """
    Iterate over cache samples in decoder-sized batches.

    The physical shard size no longer limits the decoder batch
    size.

    Example:
        cache shard size = 16
        decoder batch size = 64

    This function loads samples across four shards and combines
    them into one [64, ...] decoder batch.
    """

    shard_indices = list(
        range(
            len(
                cache.files
            )
        )
    )

    rng = random.Random(
        seed
    )

    if shuffle:
        rng.shuffle(
            shard_indices
        )

    feature_buffer = []
    target_buffer = []

    buffered_samples = 0

    for shard_index in shard_indices:

        shard = cache.load_shard(
            shard_index
        )

        features = shard[
            "features"
        ]

        targets = shard[
            "targets"
        ]

        # ----------------------------------------------------
        # Shuffle samples inside each shard for training.
        # ----------------------------------------------------

        if shuffle:

            permutation = torch.randperm(
                features.shape[0],
                generator=torch.Generator().manual_seed(
                    seed
                    + shard_index
                ),
            )

            features = features[
                permutation
            ]

            targets = targets[
                permutation
            ]

        shard_position = 0
        shard_samples = (
            features.shape[0]
        )

        # ----------------------------------------------------
        # Fill decoder-sized batches from one or more shards.
        # ----------------------------------------------------

        while (
            shard_position
            < shard_samples
        ):

            needed = (
                batch_size
                - buffered_samples
            )

            available = (
                shard_samples
                - shard_position
            )

            take = min(
                needed,
                available,
            )

            feature_piece = (
                features[
                    shard_position:
                    shard_position
                    + take
                ]
            )

            target_piece = (
                targets[
                    shard_position:
                    shard_position
                    + take
                ]
            )

            feature_buffer.append(
                feature_piece
            )

            target_buffer.append(
                target_piece
            )

            buffered_samples += take

            shard_position += take

            # ------------------------------------------------
            # We now have a full decoder batch.
            # ------------------------------------------------

            if (
                buffered_samples
                == batch_size
            ):

                if (
                    len(
                        feature_buffer
                    )
                    == 1
                ):
                    batch_features = (
                        feature_buffer[0]
                    )

                    batch_targets = (
                        target_buffer[0]
                    )

                else:
                    batch_features = (
                        torch.cat(
                            feature_buffer,
                            dim=0,
                        )
                    )

                    batch_targets = (
                        torch.cat(
                            target_buffer,
                            dim=0,
                        )
                    )

                yield (
                    batch_features,
                    batch_targets,
                )

                feature_buffer = []
                target_buffer = []

                buffered_samples = 0

        del shard
        del features
        del targets

    # --------------------------------------------------------
    # Final incomplete batch.
    # --------------------------------------------------------

    if buffered_samples > 0:

        if (
            len(
                feature_buffer
            )
            == 1
        ):
            batch_features = (
                feature_buffer[0]
            )

            batch_targets = (
                target_buffer[0]
            )

        else:
            batch_features = (
                torch.cat(
                    feature_buffer,
                    dim=0,
                )
            )

            batch_targets = (
                torch.cat(
                    target_buffer,
                    dim=0,
                )
            )

        yield (
            batch_features,
            batch_targets,
        )


# ============================================================
# Decoder helpers
# ============================================================

def decoder_forward(
    decoder,
    features,
    bf16_supported,
):
    if bf16_supported:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            return decoder(
                features
            )

    return decoder(
        features.float()
    )


def denormalize(
    tensor,
):
    mean = tensor.new_tensor(
        [
            0.485,
            0.456,
            0.406,
        ]
    ).view(
        1,
        3,
        1,
        1,
    )

    std = tensor.new_tensor(
        [
            0.229,
            0.224,
            0.225,
        ]
    ).view(
        1,
        3,
        1,
        1,
    )

    return (
        tensor
        * std
        + mean
    ).clamp(
        0,
        1,
    )


def save_preview_grid(
    decoder,
    cache,
    indices,
    output,
    filename,
    device,
    bf16_supported,
):
    decoder.eval()

    rows = []

    with torch.no_grad():

        for index in indices:

            features, target = (
                cache.get_sample(
                    index
                )
            )

            features = (
                features
                .unsqueeze(0)
                .to(device)
            )

            target = (
                target
                .unsqueeze(0)
                .to(device)
            )

            prediction = (
                decoder_forward(
                    decoder,
                    features,
                    bf16_supported,
                )
            )

            original = denormalize(
                target.float()
            )

            reconstruction = (
                denormalize(
                    prediction.float()
                )
            )

            row = torch.cat(
                [
                    original,
                    reconstruction,
                ],
                dim=3,
            )

            rows.append(
                row.cpu()
            )

    grid = torch.cat(
        rows,
        dim=0,
    )

    path = (
        output
        / filename
    )

    save_image(
        grid,
        path,
        nrow=1,
        padding=4,
    )

    print(
        f"Saved preview grid: "
        f"{path}",
        flush=True,
    )


# ============================================================
# Evaluation
# ============================================================

def report_progress(split, epoch, batch, batches, count, started, loss, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"{split} epoch={epoch} batch={batch}/{batches} mse={loss:.6f} | "
          f"{batch / max(elapsed, 1e-9):.2f} batches/s | {count / max(elapsed, 1e-9):.2f} frames/s | "
          f"{elapsed / batch:.3f} s/batch elapsed={elapsed:.1f}s", flush=True)


def evaluate_decoder(
    decoder,
    cache,
    args,
    device,
    bf16_supported,
    split="val",
    epoch=0,
):
    decoder.eval()

    native_total = 0.0
    common_total = 0.0
    count = 0
    batch_index = 0
    batches = (len(cache) + args.batch_size - 1) // args.batch_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.no_grad():

        for (
            features,
            targets,
        ) in iter_cache_batches(
            cache,
            args.batch_size,
            shuffle=False,
            seed=0,
        ):

            features = features.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            prediction = (
                decoder_forward(
                    decoder,
                    features,
                    bf16_supported,
                )
            )

            native_loss = (
                F.mse_loss(
                    prediction.float(),
                    targets.float(),
                )
            )

            prediction_256 = (
                F.interpolate(
                    prediction.float(),
                    size=(
                        256,
                        256,
                    ),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            )

            targets_256 = (
                F.interpolate(
                    targets.float(),
                    size=(
                        256,
                        256,
                    ),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            )

            common_loss = (
                F.mse_loss(
                    prediction_256,
                    targets_256,
                )
            )

            current = (
                targets.shape[0]
            )

            native_total += (
                native_loss.item()
                * current
            )

            common_total += (
                common_loss.item()
                * current
            )

            count += current
            batch_index += 1
            if batch_index == 1 or batch_index % args.log_every == 0 or batch_index == batches:
                report_progress(split, epoch, batch_index, batches, count, started, native_loss.item(), device)

    return (
        native_total
        / count,
        common_total
        / count,
    )


# ============================================================
# Decoder training
# ============================================================

def train_decoder(
    model_key,
    config,
    train_cache_dir,
    val_cache_dir,
    args,
    decoder_device,
    bf16_supported,
):
    output = (
        Path(
            args.output
        )
        / model_key
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        output
        / "latest.pt"
    )

    best_path = (
        output
        / "best_decoder.pt"
    )
    if latest_path.exists() and not args.resume:
        raise FileExistsError(f"Existing run: {output}. Use a new output directory or --resume.")


    train_cache = (
        ShardedFeatureCache(
            train_cache_dir
        )
    )

    val_cache = (
        ShardedFeatureCache(
            val_cache_dir
        )
    )

    preview_indices = (
        make_preview_indices(
            len(
                val_cache
            ),
            args.preview_count,
        )
    )

    first_features, _ = (
        val_cache.get_sample(
            preview_indices[0]
        )
    )

    feature_dim = (
        first_features.shape[-1]
    )

    decoder = ViTDecoder(
        in_dim=feature_dim,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        img_size=config[
            "img_size"
        ],
        out_channels=3,
    ).to(
        decoder_device
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    decoder_params = sum(
        parameter.numel()
        for parameter in (
            decoder.parameters()
        )
    )

    print()
    print("=" * 70)

    print(
        f"Training decoder for "
        f"{config['display_name']}"
    )

    print(
        f"Train samples: "
        f"{len(train_cache):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_cache):,}"
    )

    print(
        f"Feature shape: "
        f"{tuple(first_features.shape)}"
    )

    print(
        f"Decoder parameters: "
        f"{decoder_params:,}"
    )

    print(
        f"Cache shard size: "
        f"{train_cache.shard_sizes[0]}"
    )

    print(
        f"Decoder batch size: "
        f"{args.batch_size}"
    )

    print(
        f"Expected batches/epoch: "
        f"{(
            len(train_cache)
            + args.batch_size
            - 1
        ) // args.batch_size}"
    )

    print("=" * 70)

    start_epoch = 0
    global_step = 0

    best_val = float(
        "inf"
    )

    best_common = float(
        "inf"
    )

    if (
        args.resume
        and latest_path.exists()
    ):

        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=True,
        )

        current_identities = [read_manifest(train_cache_dir)["identity"], read_manifest(val_cache_dir)["identity"]]
        if checkpoint.get("data_identities") != current_identities:
            raise ValueError("Resume data/cache identity changed. Use a new output directory for the filtered dataset.")

        decoder.load_state_dict(
            checkpoint[
                "decoder_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            checkpoint[
                "epoch"
            ]
        )

        global_step = (
            checkpoint[
                "global_step"
            ]
        )

        best_val = (
            checkpoint[
                "best_val"
            ]
        )

        best_common = (
            checkpoint.get(
                "best_common",
                float("inf"),
            )
        )

        for state in (
            optimizer.state.values()
        ):

            for (
                key,
                value,
            ) in state.items():

                if torch.is_tensor(
                    value
                ):
                    state[key] = (
                        value.to(
                            decoder_device
                        )
                    )

    log_path = (
        output
        / "losses.csv"
    )

    append = (
        args.resume
        and start_epoch > 0
        and log_path.exists()
    )

    with log_path.open(
        "a"
        if append
        else "w",
        newline="",
    ) as log_file:

        writer = csv.writer(
            log_file
        )

        if not append:

            writer.writerow(
                [
                    "epoch",
                    "global_step",
                    "split",
                    "native_mse",
                    "common_256_mse",
                    "grad_norm",
                ]
            )

        for epoch in range(
            start_epoch,
            args.epochs,
        ):

            epoch_number = (
                epoch + 1
            )

            reset_gpu_memory_stats()

            print()
            print(
                f"{config['display_name']} | "
                f"Epoch "
                f"{epoch_number}/"
                f"{args.epochs}",
                flush=True,
            )

            decoder.train()

            total = 0.0
            count = 0
            batch_index = 0
            batches = (len(train_cache) + args.batch_size - 1) // args.batch_size
            if decoder_device.type == "cuda":
                torch.cuda.synchronize(decoder_device)
            started = time.perf_counter()

            for (
                features,
                targets,
            ) in iter_cache_batches(
                train_cache,
                args.batch_size,
                shuffle=True,
                seed=(
                    42
                    + epoch_number
                ),
            ):

                batch_index += 1

                # Helpful confirmation when tuning.
                if (
                    batch_index == 1
                ):
                    print(
                        f"Actual decoder batch shape: "
                        f"{tuple(features.shape)}",
                        flush=True,
                    )

                features = features.to(
                    decoder_device,
                    non_blocking=True,
                )

                targets = targets.to(
                    decoder_device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = (
                    decoder_forward(
                        decoder,
                        features,
                        bf16_supported,
                    )
                )

                check_finite(
                    prediction,
                    "Decoder prediction",
                )

                loss = F.mse_loss(
                    prediction.float(),
                    targets.float(),
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Non-finite loss."
                    )

                loss.backward()

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        decoder.parameters(),
                        max_norm=(
                            args.max_grad_norm
                        ),
                        error_if_nonfinite=True,
                    )
                )

                optimizer.step()

                global_step += 1

                current = (
                    targets.shape[0]
                )

                total += (
                    loss.item()
                    * current
                )

                count += current

                if (
                    batch_index == 1
                    or batch_index % args.log_every == 0
                    or batch_index == batches
                ):

                    report_progress("train", epoch_number, batch_index, batches,
                                    count, started, loss.item(), decoder_device)

                    writer.writerow(
                        [
                            epoch_number,
                            global_step,
                            "train_step",
                            loss.item(),
                            "",
                            grad_norm.item(),
                        ]
                    )

                    log_file.flush()

            train_mse = (
                total
                / count
            )

            val_mse, val_common = (
                evaluate_decoder(
                    decoder,
                    val_cache,
                    args,
                    decoder_device,
                    bf16_supported,
                    epoch=epoch_number,
                )
            )

            improved = (
                val_mse
                < best_val
            )

            if improved:
                best_val = (
                    val_mse
                )

            best_common = min(
                best_common,
                val_common,
            )

            print()
            print(
                f"Train MSE: "
                f"{train_mse:.6f}"
            )

            print(
                f"Validation MSE: "
                f"{val_mse:.6f}"
            )

            print(
                f"Validation common-256: "
                f"{val_common:.6f}",
                flush=True,
            )


            writer.writerow(
                [
                    epoch_number,
                    global_step,
                    "train_epoch",
                    train_mse,
                    "",
                    "",
                ]
            )

            writer.writerow(
                [
                    epoch_number,
                    global_step,
                    "validation",
                    val_mse,
                    val_common,
                    "",
                ]
            )

            log_file.flush()

            checkpoint = {
                "data_identities": [read_manifest(train_cache_dir)["identity"], read_manifest(val_cache_dir)["identity"]],
                "model_key": (
                    model_key
                ),
                "epoch": (
                    epoch_number
                ),
                "global_step": (
                    global_step
                ),
                "best_val": (
                    best_val
                ),
                "best_common": (
                    best_common
                ),
                "decoder_state_dict": (
                    decoder.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "feature_dim": (
                    feature_dim
                ),
                "img_size": (
                    config[
                        "img_size"
                    ]
                ),
                "config": (
                    vars(args)
                ),
            }

            temporary_path = (
                output
                / "latest.tmp"
            )

            torch.save(
                checkpoint,
                temporary_path,
            )

            temporary_path.replace(
                latest_path
            )

            if improved:

                torch.save(
                    {
                        "model_key": (
                            model_key
                        ),
                        "epoch": (
                            epoch_number
                        ),
                        "val_mse": (
                            val_mse
                        ),
                        "common_256_mse": (
                            val_common
                        ),
                        "feature_dim": (
                            feature_dim
                        ),
                        "img_size": (
                            config[
                                "img_size"
                            ]
                        ),
                        "decoder_state_dict": (
                            decoder.state_dict()
                        ),
                        "config": (
                            vars(args)
                        ),
                    },
                    best_path,
                )

                print(
                    "Saved new best decoder.",
                    flush=True,
                )

            save_preview_grid(
                decoder,
                val_cache,
                preview_indices,
                output,
                (
                    f"validation_epoch_"
                    f"{epoch_number:03d}.png"
                ),
                decoder_device,
                bf16_supported,
            )

            peak = (f"{torch.cuda.max_memory_allocated(decoder_device) / 1024**3:.2f} GiB"
                    if decoder_device.type == "cuda" else "n/a (CPU)")
            print(f"Epoch {epoch_number}: train={train_mse:.6f} val={val_mse:.6f} "
                  f"common256={val_common:.6f} | max_gpu_memory={peak}", flush=True)

    return (
        decoder,
        best_path,
        best_val,
        best_common,
    )


# ============================================================
# Test
# ============================================================

def evaluate_test(
    model_key,
    config,
    encoder,
    test_dataset,
    decoder,
    best_path,
    args,
    encoder_device,
    decoder_device,
    bf16_supported,
):
    print()
    print(
        "Caching TEST features only now, "
        "after training is complete.",
        flush=True,
    )

    test_cache_dir = (
        cache_split(
            "test",
            test_dataset,
            encoder,
            model_key,
            config,
            args,
            encoder_device,
            bf16_supported,
        )
    )

    test_cache = (
        ShardedFeatureCache(
            test_cache_dir
        )
    )

    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    decoder.load_state_dict(
        best[
            "decoder_state_dict"
        ]
    )

    decoder = (
        decoder
        .to(
            decoder_device
        )
        .eval()
    )

    reset_gpu_memory_stats()

    test_mse, test_common = (
        evaluate_decoder(
            decoder,
            test_cache,
            args,
            decoder_device,
            bf16_supported,
            split="test",
        )
    )

    print()
    print(
        f"FINAL TEST: "
        f"{config['display_name']}"
    )

    print(
        f"Test native MSE: "
        f"{test_mse:.6f}"
    )

    print(
        f"Test common-256 MSE: "
        f"{test_common:.6f}",
        flush=True,
    )

    print_gpu_memory(
        f"after "
        f"{model_key} "
        f"test evaluation"
    )

    output = (
        Path(
            args.output
        )
        / model_key
    )

    preview_indices = (
        make_preview_indices(
            len(
                test_cache
            ),
            args.preview_count,
        )
    )

    save_preview_grid(
        decoder,
        test_cache,
        preview_indices,
        output,
        "test_reconstructions.png",
        decoder_device,
        bf16_supported,
    )

    test_results_path = (
        output
        / "test_results.csv"
    )

    with test_results_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "model",
                "test_native_mse",
                "test_common_256_mse",
            ]
        )

        writer.writerow(
            [
                model_key,
                test_mse,
                test_common,
            ]
        )

    return (
        test_mse,
        test_common,
    )


# ============================================================
# One model
# ============================================================

def process_model(
    model_key,
    args,
    train_dataset,
    val_dataset,
    test_dataset,
    encoder_device,
    decoder_device,
    bf16_supported,
):
    config = VJEPA_MODELS[
        model_key
    ]

    encoder = load_encoder(
        model_key,
        args,
        encoder_device,
    )

    train_cache = cache_split(
        "train",
        train_dataset,
        encoder,
        model_key,
        config,
        args,
        encoder_device,
        bf16_supported,
    )

    val_cache = cache_split(
        "val",
        val_dataset,
        encoder,
        model_key,
        config,
        args,
        encoder_device,
        bf16_supported,
    )

    (
        decoder,
        best_path,
        best_val,
        best_common,
    ) = train_decoder(
        model_key,
        config,
        train_cache,
        val_cache,
        args,
        decoder_device,
        bf16_supported,
    )

    (
        test_mse,
        test_common,
    ) = evaluate_test(
        model_key,
        config,
        encoder,
        test_dataset,
        decoder,
        best_path,
        args,
        encoder_device,
        decoder_device,
        bf16_supported,
    )

    del encoder
    del decoder

    clear_cuda()

    return {
        "model": (
            model_key
        ),
        "display_name": (
            config[
                "display_name"
            ]
        ),
        "best_val_mse": (
            best_val
        ),
        "best_val_common_256": (
            best_common
        ),
        "test_mse": (
            test_mse
        ),
        "test_common_256": (
            test_common
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "vjepa2",
            "vjepa2_1",
            "both",
        ],
        default="vjepa2_1",
    )

    parser.add_argument(
        "--train-dir",
        default="../data/train",
    )

    parser.add_argument(
        "--val-dir",
        default="../data/val",
    )

    parser.add_argument(
        "--test-dir",
        default="../data/test",
    )

    parser.add_argument(
        "--output",
        default=(
            "../outputs/"
            "decoder_compare"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        default=(
            "../outputs/"
            "feature_cache"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        default=(
            "../checkpoints/"
            "vjepa"
        ),
    )

    parser.add_argument(
        "--vjepa-repo",
        default=".",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    # --------------------------------------------------------
    # Decoder batch size.
    #
    # NOW independent of cache shard size.
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    # --------------------------------------------------------
    # Encoder batch size / physical cache shard size.
    # --------------------------------------------------------

    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--encoder-device",
        default="cuda:0",
    )

    parser.add_argument(
        "--decoder-device",
        default="cuda:1",
    )

    parser.add_argument(
        "--cache-dtype",
        choices=[
            "bfloat16",
            "float16",
            "float32",
        ],
        default="bfloat16",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument("--gray-threshold", type=int, default=32)
    parser.add_argument("--min-gray-fraction", type=float, default=0.02)
    parser.add_argument("--manifest-dir", type=Path, default=Path("../outputs/frame_manifests"))
    args = parser.parse_args()
    if args.frame_stride < 1 or not 1 <= args.gray_threshold <= 255 or not 0 <= args.min_gray_fraction <= 1:
        parser.error("Invalid frame stride or image quality thresholds")


    # ========================================================
    # Model selection
    # ========================================================

    if (
        args.model
        == "both"
    ):
        selected_models = [
            "vjepa2",
            "vjepa2_1",
        ]

    else:
        selected_models = [
            args.model
        ]

    # ========================================================
    # Download only
    # ========================================================

    if args.download_only:

        for model_key in (
            selected_models
        ):

            download_checkpoint(
                model_key,
                args.checkpoint_dir,
            )

        return

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    encoder_device = torch.device(
        args.encoder_device
    )

    decoder_device = torch.device(
        args.decoder_device
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA devices visible: "
        f"{torch.cuda.device_count()}"
    )

    print(
        f"Encoder: "
        f"{encoder_device} | "
        f"{torch.cuda.get_device_name(encoder_device)}"
    )

    print(
        f"Decoder: "
        f"{decoder_device} | "
        f"{torch.cuda.get_device_name(decoder_device)}"
    )

    bf16_supported = (
        torch.cuda.is_bf16_supported()
    )

    print(
        f"BF16 supported: "
        f"{bf16_supported}",
        flush=True,
    )

    torch.manual_seed(
        42
    )

    random.seed(
        42
    )

    # ========================================================
    # Datasets
    # ========================================================

    train_dataset = UltrasoundFrames(
        args.train_dir,
        frame_stride=(
            args.frame_stride
        ),
        manifest_path=args.manifest_dir / "train.json",
        gray_threshold=args.gray_threshold,
        min_gray_fraction=args.min_gray_fraction,
    )

    val_dataset = UltrasoundFrames(
        args.val_dir,
        frame_stride=(
            args.frame_stride
        ),
        manifest_path=args.manifest_dir / "val.json",
        gray_threshold=args.gray_threshold,
        min_gray_fraction=args.min_gray_fraction,
    )

    test_dataset = UltrasoundFrames(
        args.test_dir,
        frame_stride=(
            args.frame_stride
        ),
        manifest_path=args.manifest_dir / "test.json",
        gray_threshold=args.gray_threshold,
        min_gray_fraction=args.min_gray_fraction,
    )

    if len(
        train_dataset
    ) == 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(
        val_dataset
    ) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    if len(
        test_dataset
    ) == 0:
        raise RuntimeError(
            "Test dataset is empty."
        )

    print()

    print(
        f"Train: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Val: "
        f"{len(val_dataset):,}"
    )

    print(
        f"Test: "
        f"{len(test_dataset):,}",
        flush=True,
    )

    # ========================================================
    # Experiments
    # ========================================================

    results = []

    for model_key in (
        selected_models
    ):

        result = process_model(
            model_key,
            args,
            train_dataset,
            val_dataset,
            test_dataset,
            encoder_device,
            decoder_device,
            bf16_supported,
        )

        results.append(
            result
        )

    # ========================================================
    # Comparison
    # ========================================================

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        output
        / "comparison.csv"
    )

    with comparison_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "model",
                "display_name",
                "best_val_mse",
                "best_val_common_256",
                "test_mse",
                "test_common_256",
            ]
        )

        for result in (
            results
        ):

            writer.writerow(
                [
                    result[
                        "model"
                    ],
                    result[
                        "display_name"
                    ],
                    result[
                        "best_val_mse"
                    ],
                    result[
                        "best_val_common_256"
                    ],
                    result[
                        "test_mse"
                    ],
                    result[
                        "test_common_256"
                    ],
                ]
            )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for result in results:

        print()

        print(
            result[
                "display_name"
            ]
        )

        print(
            f"  Best validation MSE: "
            f"{result['best_val_mse']:.6f}"
        )

        print(
            f"  Test MSE: "
            f"{result['test_mse']:.6f}"
        )

        print(
            f"  Test common-256: "
            f"{result['test_common_256']:.6f}"
        )

    print()

    print(
        f"Comparison saved to: "
        f"{comparison_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()