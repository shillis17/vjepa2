from pathlib import Path

import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UltrasoundFrames(Dataset):
    def __init__(self, data_dir, frame_stride=15):
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize(
                (384, 384),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        videos = sorted(Path(data_dir).rglob("ultrasound_bmode.mp4"))

        for video in videos:
            reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
            self.samples.extend(
                (str(video), index)
                for index in range(0, len(reader), frame_stride)
            )
            del reader

        print("Videos:", len(videos))
        print("Sampled frames:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, frame_index = self.samples[index]
        reader = VideoReader(video, ctx=cpu(0), num_threads=1)
        image = Image.fromarray(reader[frame_index].asnumpy()).convert("RGB")
        return self.transform(image)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = UltrasoundFrames("../data/train")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    frames = next(iter(loader))
    print("Batch shape:", frames.shape)
    print("All values finite:", torch.isfinite(frames).all().item())