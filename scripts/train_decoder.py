import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import UltrasoundFrames
from vjepa2_decoder.models.vit_decoder import ViTDecoder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        default="../data/train",
    )
    parser.add_argument(
        "--val-dir",
        default="../data/val",
    )
    parser.add_argument(
        "--output",
        default="outputs/decoder_v21",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=15,
    )

    # RTX 3090 #0 = frozen V-JEPA2 encoder
    parser.add_argument(
        "--encoder-device",
        default="cuda:0",
    )

    # RTX 3090 #1 = trainable decoder
    parser.add_argument(
        "--decoder-device",
        default="cuda:1",
    )

    parser.add_argument(
        "--vjepa-repo",
        default="/home/seth/World_Model/vjepa2",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # CUDA checks
    # ------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This configuration expects at least two CUDA GPUs."
        )

    torch.manual_seed(42)

    encoder_device = torch.device(args.encoder_device)
    decoder_device = torch.device(args.decoder_device)

    print(
        f"PyTorch version: {torch.__version__}",
        flush=True,
    )

    print(
        f"CUDA devices available: {torch.cuda.device_count()}",
        flush=True,
    )

    print(
        f"Encoder GPU: "
        f"{torch.cuda.get_device_name(encoder_device)} "
        f"({encoder_device})",
        flush=True,
    )

    print(
        f"Decoder GPU: "
        f"{torch.cuda.get_device_name(decoder_device)} "
        f"({decoder_device})",
        flush=True,
    )

    bf16_supported = torch.cuda.is_bf16_supported()

    print(
        f"BF16 supported: {bf16_supported}",
        flush=True,
    )

    if bf16_supported:
        print(
            "Using BF16 mixed precision.",
            flush=True,
        )
    else:
        print(
            "BF16 unavailable. Falling back to FP32.",
            flush=True,
        )

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output / "latest.pt"
    )

    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError(
            f"{checkpoint_path} already exists. "
            "Use --resume or choose a different --output directory."
        )

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------

    train_dataset = UltrasoundFrames(
        args.train_dir,
        frame_stride=args.frame_stride,
    )

    val_dataset = UltrasoundFrames(
        args.val_dir,
        frame_stride=args.frame_stride,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    print(
        f"Training sampled frames: {len(train_dataset)}",
        flush=True,
    )

    print(
        f"Validation sampled frames: {len(val_dataset)}",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Same validation image every epoch so improvement is visible.
    preview_frame = (
        val_dataset[0]
        .unsqueeze(0)
    )

    # ------------------------------------------------------------
    # Load V-JEPA2 encoder locally
    # ------------------------------------------------------------

    vjepa_repo = (
        Path(args.vjepa_repo)
        .resolve()
    )

    if not vjepa_repo.exists():
        raise RuntimeError(
            f"V-JEPA2 repo does not exist: {vjepa_repo}"
        )

    hubconf_path = (
        vjepa_repo / "hubconf.py"
    )

    if not hubconf_path.exists():
        raise RuntimeError(
            f"hubconf.py does not exist: {hubconf_path}"
        )

    print(
        f"Loading V-JEPA2 from local repo: {vjepa_repo}",
        flush=True,
    )

    encoder, predictor = torch.hub.load(
        str(vjepa_repo),
        "vjepa2_1_vit_large_384",
        source="local",
    )

    del predictor

    encoder = (
        encoder
        .eval()
        .requires_grad_(False)
        .to(encoder_device)
    )

    print(
        "V-JEPA2 encoder loaded.",
        flush=True,
    )

    # ------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------

    decoder = ViTDecoder(
        in_dim=1024,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        img_size=384,
        out_channels=3,
    ).to(
        decoder_device
    )

    total_params = sum(
        p.numel()
        for p in decoder.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in decoder.parameters()
        if p.requires_grad
    )

    print(
        f"Decoder parameters: {total_params:,}",
        flush=True,
    )

    print(
        f"Trainable parameters: {trainable_params:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )

    # ------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    # ------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------

    if args.resume:
        if not checkpoint_path.exists():
            raise RuntimeError(
                "--resume was specified, but "
                f"{checkpoint_path} does not exist."
            )

        print(
            f"Loading checkpoint: {checkpoint_path}",
            flush=True,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

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

        start_epoch = checkpoint["epoch"]
        global_step = checkpoint[
            "global_step"
        ]
        best_val = checkpoint[
            "best_val"
        ]

        # Move optimizer tensors back to decoder GPU.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(
                        decoder_device
                    )

        print(
            f"Resuming after epoch {start_epoch}",
            flush=True,
        )

        print(
            f"Global step: {global_step}",
            flush=True,
        )

        print(
            f"Best validation MSE: {best_val:.6f}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Helper: encoder forward
    # ------------------------------------------------------------

    def encode(frames):
        inputs = frames.to(
            encoder_device,
            non_blocking=True,
        )

        # Dataset:
        #
        # [B, C, H, W]
        #
        # V-JEPA2:
        #
        # [B, C, T, H, W]
        #
        # We use one frame, so T = 1.
        inputs = inputs.unsqueeze(2)

        with torch.no_grad():

            if bf16_supported:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    features = encoder(
                        inputs
                    )
            else:
                features = encoder(
                    inputs.float()
                )

        # Move encoder output from GPU 0
        # to decoder on GPU 1.
        features = features.to(
            decoder_device,
            non_blocking=True,
        )

        return features

    # ------------------------------------------------------------
    # Helper: decoder forward
    # ------------------------------------------------------------

    def decode(features):
        if bf16_supported:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                prediction = decoder(
                    features
                )
        else:
            prediction = decoder(
                features.float()
            )

        return prediction

    # ------------------------------------------------------------
    # Helper: numerical checks
    # ------------------------------------------------------------

    def check_finite(
        tensor,
        name,
        epoch=None,
        batch=None,
    ):
        if not torch.isfinite(
            tensor
        ).all():
            location = ""

            if epoch is not None:
                location += (
                    f" epoch={epoch}"
                )

            if batch is not None:
                location += (
                    f" batch={batch}"
                )

            raise RuntimeError(
                f"{name} contains NaN or Inf."
                f"{location}"
            )

    # ------------------------------------------------------------
    # Helper: save reconstruction preview
    # ------------------------------------------------------------

    def save_preview(epoch):
        decoder.eval()

        features = encode(
            preview_frame
        )

        check_finite(
            features,
            "Preview encoder features",
            epoch=epoch,
        )

        with torch.no_grad():
            prediction = decode(
                features
            )

        check_finite(
            prediction,
            "Preview decoder prediction",
            epoch=epoch,
        )

        target = preview_frame.to(
            decoder_device,
            non_blocking=True,
        )

        mean = target.new_tensor(
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

        std = target.new_tensor(
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

        original = (
            target.float()
            * std
            + mean
        ).clamp(
            0,
            1,
        )

        reconstructed = (
            prediction.float()
            * std
            + mean
        ).clamp(
            0,
            1,
        )

        # Left  = original
        # Right = reconstruction
        comparison = torch.cat(
            [
                original,
                reconstructed,
            ],
            dim=3,
        )

        preview_path = (
            output
            / f"validation_epoch_{epoch:03d}.png"
        )

        save_image(
            comparison.cpu(),
            preview_path,
        )

        print(
            f"Saved preview: {preview_path}",
            flush=True,
        )

    # ------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------

    log_path = (
        output / "losses.csv"
    )

    append = (
        args.resume
        and log_path.exists()
    )

    with log_path.open(
        "a" if append else "w",
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
                    "mse",
                    "grad_norm",
                ]
            )

        # ========================================================
        # Training epochs
        # ========================================================

        for epoch in range(
            start_epoch,
            args.epochs,
        ):

            epoch_number = (
                epoch + 1
            )

            print(
                "",
                flush=True,
            )

            print(
                f"Starting epoch "
                f"{epoch_number}/{args.epochs}",
                flush=True,
            )

            decoder.train()

            train_total = 0.0
            train_count = 0

            # ----------------------------------------------------
            # Training batches
            # ----------------------------------------------------

            for (
                batch_index,
                frames,
            ) in enumerate(
                train_loader
            ):

                batch_number = (
                    batch_index + 1
                )

                # ------------------------------
                # Frozen encoder
                # ------------------------------

                features = encode(
                    frames
                )

                check_finite(
                    features,
                    "Encoder features",
                    epoch=epoch_number,
                    batch=batch_number,
                )

                # ------------------------------
                # Targets
                # ------------------------------

                targets = frames.to(
                    decoder_device,
                    non_blocking=True,
                )

                check_finite(
                    targets,
                    "Targets",
                    epoch=epoch_number,
                    batch=batch_number,
                )

                # ------------------------------
                # Decoder
                # ------------------------------

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = decode(
                    features
                )

                check_finite(
                    prediction,
                    "Decoder prediction",
                    epoch=epoch_number,
                    batch=batch_number,
                )

                # ------------------------------
                # FP32 loss
                # ------------------------------

                loss = F.mse_loss(
                    prediction.float(),
                    targets.float(),
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Training loss is "
                        "NaN or Inf at "
                        f"epoch {epoch_number}, "
                        f"batch {batch_number}. "
                        f"Loss={loss.item()}"
                    )

                # ------------------------------
                # Backward
                # ------------------------------

                loss.backward()

                # ------------------------------
                # Gradient clipping
                # ------------------------------

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(),
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )
                )

                if not torch.isfinite(
                    grad_norm
                ):
                    raise RuntimeError(
                        "Gradient norm became "
                        "NaN or Inf at "
                        f"epoch {epoch_number}, "
                        f"batch {batch_number}."
                    )

                # ------------------------------
                # Optimizer
                # ------------------------------

                optimizer.step()

                global_step += 1

                current_batch_size = (
                    frames.shape[0]
                )

                train_total += (
                    loss.item()
                    * current_batch_size
                )

                train_count += (
                    current_batch_size
                )

                # ------------------------------
                # Logging
                # ------------------------------

                if (
                    batch_index == 0
                    or global_step % 50 == 0
                ):

                    print(
                        f"Epoch "
                        f"{epoch_number}/{args.epochs} | "
                        f"Batch "
                        f"{batch_number}/{len(train_loader)} | "
                        f"Step {global_step} | "
                        f"MSE {loss.item():.6f} | "
                        f"GradNorm {grad_norm.item():.6f}",
                        flush=True,
                    )

                    writer.writerow(
                        [
                            epoch_number,
                            global_step,
                            "train_step",
                            loss.item(),
                            grad_norm.item(),
                        ]
                    )

                    log_file.flush()

            train_mse = (
                train_total
                / train_count
            )

            # ====================================================
            # Validation
            # ====================================================

            decoder.eval()

            val_total = 0.0
            val_count = 0

            with torch.no_grad():

                for (
                    batch_index,
                    frames,
                ) in enumerate(
                    val_loader
                ):

                    batch_number = (
                        batch_index + 1
                    )

                    features = encode(
                        frames
                    )

                    check_finite(
                        features,
                        "Validation encoder features",
                        epoch=epoch_number,
                        batch=batch_number,
                    )

                    targets = frames.to(
                        decoder_device,
                        non_blocking=True,
                    )

                    prediction = decode(
                        features
                    )

                    check_finite(
                        prediction,
                        "Validation decoder prediction",
                        epoch=epoch_number,
                        batch=batch_number,
                    )

                    loss = F.mse_loss(
                        prediction.float(),
                        targets.float(),
                    )

                    if not torch.isfinite(
                        loss
                    ):
                        raise RuntimeError(
                            "Validation loss is "
                            "NaN or Inf at "
                            f"epoch {epoch_number}, "
                            f"batch {batch_number}."
                        )

                    current_batch_size = (
                        frames.shape[0]
                    )

                    val_total += (
                        loss.item()
                        * current_batch_size
                    )

                    val_count += (
                        current_batch_size
                    )

            val_mse = (
                val_total
                / val_count
            )

            improved = (
                val_mse
                < best_val
            )

            if improved:
                best_val = val_mse

            # ----------------------------------------------------
            # Epoch logging
            # ----------------------------------------------------

            writer.writerow(
                [
                    epoch_number,
                    global_step,
                    "train_epoch",
                    train_mse,
                    "",
                ]
            )

            writer.writerow(
                [
                    epoch_number,
                    global_step,
                    "validation",
                    val_mse,
                    "",
                ]
            )

            log_file.flush()

            print(
                f"Epoch {epoch_number} complete | "
                f"train MSE={train_mse:.6f} | "
                f"val MSE={val_mse:.6f} | "
                f"best val MSE={best_val:.6f}",
                flush=True,
            )

            # ====================================================
            # Checkpoint
            # ====================================================

            checkpoint = {
                "epoch": epoch_number,
                "global_step": global_step,
                "best_val": best_val,
                "decoder_state_dict": (
                    decoder.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "config": vars(args),
                "vjepa_repo": str(
                    vjepa_repo
                ),
                "bf16_supported": (
                    bf16_supported
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
                checkpoint_path
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_path}",
                flush=True,
            )

            # ====================================================
            # Best decoder
            # ====================================================

            if improved:
                best_path = (
                    output
                    / "best_decoder.pt"
                )

                torch.save(
                    {
                        "decoder_state_dict": (
                            decoder.state_dict()
                        ),
                        "epoch": (
                            epoch_number
                        ),
                        "val_mse": (
                            val_mse
                        ),
                        "config": (
                            vars(args)
                        ),
                        "vjepa_repo": (
                            str(vjepa_repo)
                        ),
                        "bf16_supported": (
                            bf16_supported
                        ),
                    },
                    best_path,
                )

                print(
                    f"Saved new best decoder: "
                    f"{best_path}",
                    flush=True,
                )

            # ====================================================
            # Preview
            # ====================================================

            save_preview(
                epoch_number
            )

            # ----------------------------------------------------
            # GPU memory status
            # ----------------------------------------------------

            print(
                "",
                flush=True,
            )

            print(
                "GPU memory after epoch:",
                flush=True,
            )

            for gpu_index in range(
                torch.cuda.device_count()
            ):
                allocated = (
                    torch.cuda.memory_allocated(
                        gpu_index
                    )
                    / (1024 ** 3)
                )

                reserved = (
                    torch.cuda.memory_reserved(
                        gpu_index
                    )
                    / (1024 ** 3)
                )

                print(
                    f"  GPU {gpu_index}: "
                    f"allocated={allocated:.2f} GB, "
                    f"reserved={reserved:.2f} GB",
                    flush=True,
                )

    print(
        "",
        flush=True,
    )

    print(
        "Training complete.",
        flush=True,
    )

    print(
        f"Best validation MSE: "
        f"{best_val:.6f}",
        flush=True,
    )

    print(
        f"Best decoder: "
        f"{output / 'best_decoder.pt'}",
        flush=True,
    )

    print(
        f"Validation previews: "
        f"{output}/validation_epoch_XXX.png",
        flush=True,
    )


if __name__ == "__main__":
    main()