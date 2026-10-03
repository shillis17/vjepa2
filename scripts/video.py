from pathlib import Path
from decord import VideoReader, cpu

data_dir = Path("/beegfs/home/shilli/World_Model/data/train")
video = next(data_dir.rglob("ultrasound_bmode.mp4"), None)

if video is None:
    raise SystemExit(f"No ultrasound video found in {data_dir}")

reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
frame = reader[0].asnumpy()

print("Video:", video)
print("Frame count:", len(reader))
print("First frame shape:", frame.shape)
print("PASS: video decoding works.")
