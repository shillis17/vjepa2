"""Image-only ultrasound selection; manifests never depend on tracking quality."""
import hashlib
import json
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_quality(pixels, gray_threshold=32, min_gray_fraction=.02):
    # Reject uniform frames including all-black and all-white captures.
    if int(pixels.max()) == int(pixels.min()):
        return 'blank', 0.0
    gray = pixels.astype(np.float32) @ np.array([.299, .587, .114], dtype=np.float32)
    fraction = float(np.mean(gray >= gray_threshold))
    return (None if fraction >= min_gray_fraction else 'low_gray_fraction'), fraction


class UltrasoundFrames(Dataset):
    def __init__(self, data_dir, frame_stride=1, manifest_path=None,
                 gray_threshold=32, min_gray_fraction=.02):
        if frame_stride < 1 or not 1 <= gray_threshold <= 255 or not 0 <= min_gray_fraction <= 1:
            raise ValueError('Invalid stride or gray/white pixel thresholds')
        root = Path(data_dir).resolve()
        manifest_path = Path(manifest_path) if manifest_path else root / 'frame_manifest.json'
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manual = old.get('manual_exclusions', [])
        if not isinstance(manual, list):
            raise ValueError('manual_exclusions must be a list')
        exclusions = {}
        for entry in manual:
            name = entry['video']
            intervals = exclusions.setdefault(name, [])
            for frame in entry.get('frames', []):
                if not isinstance(frame, int) or frame < 0:
                    raise ValueError('Excluded frame indices must be nonnegative integers')
                intervals.append((frame, frame))
            for start, end in entry.get('ranges', []):
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end:
                    raise ValueError('Ranges must be inclusive nonnegative integer pairs')
                intervals.append((start, end))
        videos = sorted(root.rglob('ultrasound_bmode.mp4'))
        names = {v.relative_to(root).as_posix() for v in videos}
        if set(exclusions) - names:
            raise ValueError(f'Unknown video paths in {manifest_path}: {set(exclusions) - names}')
        self.samples = []
        records = []
        settings = dict(frame_stride=frame_stride, gray_threshold=gray_threshold,
                        min_gray_fraction=min_gray_fraction, preprocessing='rgb-pil-bilinear384-imagenet-v1')
        for video in videos:
            name = video.relative_to(root).as_posix()
            reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
            accepted, rejected = [], []
            for index in range(0, len(reader), frame_stride):
                if any(a <= index <= b for a, b in exclusions.get(name, [])):
                    rejected.append(dict(frame=index, reason='manual'))
                    continue
                try:
                    pixels = reader[index].asnumpy()
                except Exception as error:
                    # Record decoder failures, but do not silently discard an entire bad session.
                    rejected.append(dict(frame=index, reason='decode_error', error=str(error)))
                    continue
                reason, fraction = frame_quality(pixels, gray_threshold, min_gray_fraction)
                if reason:
                    rejected.append(dict(frame=index, reason=reason, gray_fraction=fraction))
                else:
                    accepted.append(index)
                    self.samples.append((str(video), index))
            if not accepted:
                print(f'WARNING: no accepted sampled frames in {name}', flush=True)
            records.append(dict(video=name, sha256=file_digest(video), total_frames=len(reader),
                                accepted_frames=accepted, automatic_and_manual_exclusions=rejected))
            print(f'{name}: accepted={len(accepted)} excluded={len(rejected)}', flush=True)
            del reader
        identity = dict(settings=settings, videos=[{k: r[k] for k in ('video', 'sha256', 'accepted_frames')} for r in records])
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        report = dict(format_version=1, index_convention='zero-based video frames; inclusive ranges',
                      manual_exclusions=manual, settings=settings, fingerprint=self.fingerprint, videos=records)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(manifest_path)
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
        ])
        print(f'Videos: {len(videos)} | Accepted frames: {len(self.samples)} | Manifest: {manifest_path}', flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, frame_index = self.samples[index]
        reader = VideoReader(video, ctx=cpu(0), num_threads=1)
        image = Image.fromarray(reader[frame_index].asnumpy()).convert('RGB')
        return self.transform(image)
