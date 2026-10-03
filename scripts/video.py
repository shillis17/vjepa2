import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dataset import UltrasoundFrames
from vjepa2_decoder.models.vit_decoder import ViTDecoder

from pathlib import Path
from torchvision.utils import save_image


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")

    dataset = UltrasoundFrames("../data/train")
    loader = DataLoader(
        dataset,
        batch_size=1,  # Start small: training needs more memory.
        shuffle=True,
        num_workers=0,
    )

    image_tensor = next(iter(loader))
    targets = image_tensor.to(device)
    print("Batch shape:", targets.shape)

    # Download from Meta or reuse the cached checkpoint.
    torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        map_location="cpu",
    )

    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2",
        "vjepa2_1_vit_large_384",
    )
    del predictor

    encoder = encoder.eval().requires_grad_(False).to(device)
    print("Encoder loaded and frozen.")

    # Encode each image independently.
    inputs = targets.unsqueeze(2)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        features = encoder(inputs)

    print("Feature shape:", features.shape)

    if not torch.isfinite(features).all().item():
        raise RuntimeError("Encoder features contain invalid values.")

    # Release GPU memory occupied by the encoder for this one-step test.
    # A full training loop would manage the encoder differently.
    encoder.to("cpu")
    del encoder, inputs
    torch.cuda.empty_cache()

    decoder = ViTDecoder(
        in_dim=1024,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        img_size=384,
        out_channels=3,
    ).to(device)
    decoder.train()

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-4,
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda")

    for step in range(500):
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.float16):
            reconstruction = decoder(features)
            loss = F.mse_loss(reconstruction, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step + 1}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        scaler.step(optimizer)
        scaler.update()

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"Step {step + 1}/100 | "
                f"MSE: {loss.item():.6f} | "
                f"Gradient norm: {grad_norm.item():.4f}",
                flush=True,
            )

    print("PASS: 100 training steps completed.")

    # Reconstruct again using the final updated weights.
    decoder.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        reconstruction = decoder(features)

    # Undo ImageNet normalization for display.
    mean = targets.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = targets.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    original = (targets.float() * std + mean).clamp(0, 1)
    decoded = (reconstruction.float() * std + mean).clamp(0, 1)

    # Original on the left, reconstruction on the right.
    comparison = torch.cat([original[:1], decoded[:1]], dim=3)

    output_path = Path("outputs/reconstruction_test.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(comparison.cpu(), output_path)

    print("Saved comparison:", output_path.resolve())

if __name__ == "__main__":
    main()