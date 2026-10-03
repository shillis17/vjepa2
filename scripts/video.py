from pathlib import Path
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms

data_dir = Path("/beegfs/home/shilli/World_Model/data/train")
video = next(data_dir.rglob("ultrasound_bmode.mp4"), None)

if video is None:
    raise FileNotFoundError("No ultrasound video found.")

reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
image = Image.fromarray(reader[0].asnumpy()).convert("RGB")
image = image.resize((384, 384), Image.Resampling.BILINEAR)

normalize = transforms.Compose([
    transforms.ToTensor(),  # Converts pixel values from 0–255 to 0–1
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

image_tensor = normalize(image)

print("Tensor shape:", image_tensor.shape)
print("Tensor type:", image_tensor.dtype)
