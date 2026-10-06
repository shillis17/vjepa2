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
    """Hash file contents without retaining the file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_quality(pixels, gray_threshold=32, min_gray_fraction=0.02):
    """Classify uniform or low-signal RGB frames."""
    if int(pixels.max()) == int(pixels.min()):
        return ("blank", 0.0)
    gray = pixels.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    fraction = float(np.mean(gray >= gray_threshold))
    return (None if fraction >= min_gray_fraction else "low_gray_fraction", fraction)


def decoded_frames(reader, indices, batch_size=16):
    """Bounded batch decoding, with per-frame fallback for corrupt batches."""
    for offset in range(0, len(indices), batch_size):
        batch = indices[offset : offset + batch_size]
        try:
            pixels = reader.get_batch(list(batch)).asnumpy()
        except Exception:
            for i in batch:
                try:
                    frame = reader[i].asnumpy()
                except Exception:
                    frame = None
                yield (i, frame)
        else:
            yield from zip(batch, pixels)


class UltrasoundFrames(Dataset):

    def __init__(
        self,
        data_dir,
        frame_stride=1,
        manifest_path=None,
        gray_threshold=32,
        min_gray_fraction=0.02,
        rescan_frames=False,
    ):
        if (
            frame_stride < 1
            or not 1 <= gray_threshold <= 255
            or (not 0 <= min_gray_fraction <= 1)
        ):
            raise ValueError("Invalid stride or gray/white pixel thresholds")
        root = Path(data_dir).resolve()
        manifest_path = (
            Path(manifest_path) if manifest_path else root / "frame_manifest.json"
        )
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manual = old.get("manual_exclusions", [])
        if not isinstance(manual, list):
            raise ValueError("manual_exclusions must be a list")
        exclusions = {}
        for entry in manual:
            name = entry["video"]
            intervals = exclusions.setdefault(name, [])
            for frame in entry.get("frames", []):
                if not isinstance(frame, int) or frame < 0:
                    raise ValueError(
                        "Excluded frame indices must be nonnegative integers"
                    )
                intervals.append((frame, frame))
            for start, end in entry.get("ranges", []):
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or (not 0 <= start <= end)
                ):
                    raise ValueError(
                        "Ranges must be inclusive nonnegative integer pairs"
                    )
                intervals.append((start, end))
        videos = sorted(root.rglob("ultrasound_bmode.mp4"))
        names = {v.relative_to(root).as_posix() for v in videos}
        if set(exclusions) - names:
            raise ValueError(
                f"Unknown video paths in {manifest_path}: {set(exclusions) - names}"
            )
        self.samples = []
        records = []
        settings = dict(
            frame_stride=frame_stride,
            gray_threshold=gray_threshold,
            min_gray_fraction=min_gray_fraction,
            preprocessing="rgb-pil-bilinear384-imagenet-v1",
        )
        old_records = {r["video"]: r for r in old.get("videos", [])}
        old_exclusions = {}
        for entry in old.get("manual_exclusions", []):
            old_exclusions.setdefault(entry["video"], []).extend(
                [[i, i] for i in entry.get("frames", [])] + entry.get("ranges", [])
            )
        for video in videos:
            name = video.relative_to(root).as_posix()
            source_digest = file_digest(video)
            previous = old_records.get(name)
            if (
                not rescan_frames
                and old.get("format_version") == 2
                and previous
                and (old.get("settings") == settings)
                and (
                    previous.get("manual_ranges", old_exclusions.get(name, []))
                    == [list(r) for r in exclusions.get(name, [])]
                )
                and (previous.get("sha256") == source_digest)
            ):
                payload = {k: v for k, v in previous.items() if k != "record_checksum"}
                checksum = hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest()
                if previous.get("record_checksum") == checksum:
                    records.append(previous)
                    self.samples.extend(
                        ((str(video), i) for i in previous["accepted_frames"])
                    )
                    print(
                        f"{name}: reusing verified frame scan ({len(previous['accepted_frames'])} accepted)",
                        flush=True,
                    )
                    continue
            reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
            accepted, rejected = ([], [])
            for index, pixels in decoded_frames(
                reader, range(0, len(reader), frame_stride)
            ):
                if any((a <= index <= b for a, b in exclusions.get(name, []))):
                    rejected.append(dict(frame=index, reason="manual"))
                    continue
                if pixels is None:
                    rejected.append(dict(frame=index, reason="decode_error"))
                    continue
                reason, fraction = frame_quality(
                    pixels, gray_threshold, min_gray_fraction
                )
                if reason:
                    rejected.append(
                        dict(frame=index, reason=reason, gray_fraction=fraction)
                    )
                else:
                    accepted.append(index)
                    self.samples.append((str(video), index))
            if not accepted:
                print(f"WARNING: no accepted sampled frames in {name}", flush=True)
            records.append(
                dict(
                    video=name,
                    sha256=source_digest,
                    total_frames=len(reader),
                    accepted_frames=accepted,
                    automatic_and_manual_exclusions=rejected,
                )
            )
            print(
                f"{name}: accepted={len(accepted)} excluded={len(rejected)}", flush=True
            )
            del reader
        identity = dict(
            settings=settings,
            videos=[
                {k: r[k] for k in ("video", "sha256", "accepted_frames")}
                for r in records
            ],
        )
        self.fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        for record in records:
            record["manual_ranges"] = [
                list(r) for r in exclusions.get(record["video"], [])
            ]
            payload = {k: v for k, v in record.items() if k != "record_checksum"}
            record["record_checksum"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
        report = dict(
            format_version=2,
            index_convention="zero-based video frames; inclusive ranges",
            manual_exclusions=manual,
            settings=settings,
            fingerprint=self.fingerprint,
            videos=records,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(manifest_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        print(
            f"Videos: {len(videos)} | Accepted frames: {len(self.samples)} | Manifest: {manifest_path}",
            flush=True,
        )

    def __len__(self):
        """Return the number of selected samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Decode and normalize a selected sample."""
        video, frame_index = self.samples[index]
        reader = VideoReader(video, ctx=cpu(0), num_threads=1)
        image = Image.fromarray(reader[frame_index].asnumpy()).convert("RGB")
        return self.transform(image)
