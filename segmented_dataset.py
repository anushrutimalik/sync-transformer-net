"""
SegmentedDataset: Phase 1 Enhancement for SyncWeld-Net
- Divides videos into fixed 3-5 second segments
- Strict Real/Fake class balancing
- Synchronized audio (16kHz) and video (2 FPS) extraction
"""

import os
import torch
import cv2
import numpy as np
import subprocess
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import random


class SegmentedFakeAVCelebDataset(Dataset):
    """
    Enhanced dataset that segments videos into fixed-duration clips for
    increased training samples and better temporal artifact detection.
    """

    def __init__(
        self,
        metadata_csv_path: str,
        base_path: str,
        num_frames: int = 8,
        image_size: int = 160,
        target_sample_rate: int = 16000,
        segment_duration: float = 4.0,
        overlap: float = 0.0,
        split: str = "all",
        val_ratio: float = 0.2,
        max_samples: Optional[int] = None,
        max_segments_per_video: int = 3,
        ensure_balance: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            metadata_csv_path: Path to FakeAVCeleb meta_data.csv
            base_path: Root directory of FakeAVCeleb dataset
            num_frames: Number of frames to extract per segment
            image_size: Resolution for face crops
            target_sample_rate: Audio sample rate (16kHz for Wav2Vec2)
            segment_duration: Duration of each segment in seconds (3-5 recommended)
            overlap: Overlap between segments (0-1), 0 = no overlap
            split: 'train', 'val', or 'all'
            val_ratio: Proportion for validation split
            max_samples: Maximum total segments to use
            max_segments_per_video: Max segments to extract from each video
            ensure_balance: If True, strictly balance Real/Fake classes
            seed: Random seed for reproducibility
        """
        import csv

        random.seed(seed)
        np.random.seed(seed)

        self.base_path = base_path
        self.num_frames = num_frames
        self.image_size = image_size
        self.target_sample_rate = target_sample_rate
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.max_segments_per_video = max_segments_per_video

        all_segments = []

        with open(metadata_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_prefix = row.get("", "")
                if rel_prefix.startswith("FakeAVCeleb/"):
                    rel_prefix = rel_prefix.replace("FakeAVCeleb/", "", 1)

                video_file_name = row["path"]
                full_video_path = os.path.join(base_path, rel_prefix, video_file_name)

                if not os.path.exists(full_video_path):
                    continue

                category = row["type"]

                # Label: 0 = Real (Pristine), 1 = Fake (any manipulation)
                # RealVideo-FakeAudio should also be FAKE (audio is fake!)
                if category == "RealVideo-RealAudio":
                    label = 0
                    detailed_label = "RealVideo-RealAudio"
                else:
                    label = 1
                    # Map to detailed labels
                    if category == "FakeVideo-FakeAudio":
                        detailed_label = "FakeVideo-FakeAudio"
                    elif category == "FakeVideo-RealAudio":
                        detailed_label = "FakeVideo-RealAudio"
                    elif category == "RealVideo-FakeAudio":
                        detailed_label = "RealVideo-FakeAudio"
                    else:
                        detailed_label = category

                segments_info = self._extract_segment_boundaries(
                    full_video_path, max_segments_per_video
                )

                for seg_start, seg_end in segments_info:
                    all_segments.append(
                        {
                            "video_path": full_video_path,
                            "label": label,
                            "category": category,
                            "detailed_label": detailed_label,
                            "seg_start": seg_start,
                            "seg_end": seg_end,
                            "seg_idx": len(segments_info)
                            - segments_info.index((seg_start, seg_end))
                            - 1,
                        }
                    )

        if ensure_balance:
            all_segments = self._balance_classes(all_segments, seed)

        if split in ["train", "val"]:
            random.shuffle(all_segments)
            split_point = int(len(all_segments) * (1 - val_ratio))
            if split == "train":
                all_segments = all_segments[:split_point]
            else:
                all_segments = all_segments[split_point:]

        if max_samples is not None and len(all_segments) > max_samples:
            all_segments = all_segments[:max_samples]

        self.metadata = all_segments
        self._print_statistics()

    def _balance_classes(self, segments: List[dict], seed: int) -> List[dict]:
        """Ensure strict balance between Real and Fake classes."""
        real_segs = [s for s in segments if s["label"] == 0]
        fake_segs = [s for s in segments if s["label"] == 1]

        min_count = min(len(real_segs), len(fake_segs))

        random.seed(seed)
        real_sampled = random.sample(real_segs, min_count)
        fake_sampled = random.sample(fake_segs, min_count)

        balanced = real_sampled + fake_sampled
        random.shuffle(balanced)

        return balanced

    def _print_statistics(self):
        """Print dataset statistics for the paper."""
        real_count = sum(1 for s in self.metadata if s["label"] == 0)
        fake_count = sum(1 for s in self.metadata if s["label"] == 1)
        print(f"SegmentedDataset: {len(self.metadata)} total segments")
        print(
            f"  Real (Pristine): {real_count} ({100 * real_count / len(self.metadata):.1f}%)"
        )
        print(f"  Fake: {fake_count} ({100 * fake_count / len(self.metadata):.1f}%)")
        print(f"  Segment Duration: {self.segment_duration}s")
        print(f"  Frames per segment: {self.num_frames}")

    def _extract_segment_boundaries(
        self, video_path: str, max_segments: int
    ) -> List[Tuple[float, float]]:
        """Extract segment boundaries from video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if duration < self.segment_duration:
            return [(0.0, duration)] if duration > 0.5 else []

        step = self.segment_duration * (1 - self.overlap)
        segments = []

        start = 0.0
        while start < duration and len(segments) < max_segments:
            end = min(start + self.segment_duration, duration)
            segments.append((start, end))
            start += step

        return segments

    def _extract_segmented_frames(
        self, video_path: str, seg_start: float, seg_end: float
    ) -> torch.Tensor:
        """Extract frames from a specific video segment."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(seg_start * fps)
        end_frame = int(seg_end * fps)

        frames_in_segment = end_frame - start_frame

        if frames_in_segment <= 0:
            cap.release()
            return torch.zeros((self.num_frames, 3, self.image_size, self.image_size))

        frame_indices = np.linspace(
            start_frame, end_frame - 1, self.num_frames, dtype=int
        )
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            else:
                frames.append(
                    np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                )

        cap.release()

        frames_array = np.stack(frames)
        frames_tensor = (
            torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0
        )
        return frames_tensor

    def _extract_segmented_audio(
        self, video_path: str, seg_start: float, seg_end: float
    ) -> torch.Tensor:
        """Extract audio from a specific video segment at 16kHz."""
        try:
            try:
                import torchaudio

                waveform, sample_rate = torchaudio.load(
                    video_path,
                    frame_offset=int(seg_start * sample_rate),
                    num_frames=int((seg_end - seg_start) * sample_rate),
                )
            except Exception:
                import av

                container = av.open(video_path)
                try:
                    audio_stream = container.streams.audio[0]
                except IndexError:
                    container.close()
                    return torch.zeros(
                        self.target_sample_rate * int(self.segment_duration)
                    )

                actual_sample_rate = audio_stream.codec_context.sample_rate

                samples = []
                for frame in container.decode(audio_stream):
                    frame_pos = frame.pts * av.time_base
                    frame_time = (
                        frame_pos.num / frame_pos.den if frame_pos.den != 0 else 0
                    )

                    if frame_time >= seg_start - 0.1 and frame_time <= seg_end + 0.1:
                        samples.append(frame.to_ndarray())

                container.close()

                if len(samples) > 0:
                    waveform_np = np.concatenate(samples, axis=-1)
                    waveform = torch.from_numpy(waveform_np).float()
                    sample_rate = actual_sample_rate
                else:
                    return torch.zeros(
                        self.target_sample_rate * int(self.segment_duration)
                    )

            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            target_length = self.target_sample_rate * int(self.segment_duration)
            if waveform.shape[0] > target_length:
                waveform = waveform[:target_length]
            else:
                padding = target_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform

        except Exception as e:
            return torch.zeros(self.target_sample_rate * int(self.segment_duration))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seg_info = self.metadata[idx]

        visual_tensor = self._extract_segmented_frames(
            seg_info["video_path"], seg_info["seg_start"], seg_info["seg_end"]
        )

        audio_tensor = self._extract_segmented_audio(
            seg_info["video_path"], seg_info["seg_start"], seg_info["seg_end"]
        )

        label = torch.tensor(seg_info["label"], dtype=torch.float32)

        return visual_tensor, audio_tensor, label


class SynchronizedExtractor:
    """
    FFmpeg-based synchronized feature extractor for precise audio-video alignment.
    Extracts frames at exactly 2 FPS and audio at 16kHz.
    """

    def __init__(self, target_fps: float = 2.0, target_sr: int = 16000):
        self.target_fps = target_fps
        self.target_sr = target_sr

    def extract_with_ffmpeg(
        self, video_path: str, output_dir: str, segment_name: str
    ) -> dict:
        """
        Use FFmpeg to extract synchronized frames and audio.
        Returns paths to extracted files and metadata.
        """
        os.makedirs(output_dir, exist_ok=True)

        frames_dir = os.path.join(output_dir, f"{segment_name}_frames")
        audio_path = os.path.join(output_dir, f"{segment_name}_audio.wav")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            cmd_frames = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"fps={self.target_fps}",
                "-q:v",
                "2",
                os.path.join(frames_dir, "frame_%04d.jpg"),
                "-y",
            ]
            subprocess.run(cmd_frames, capture_output=True, check=False)

            cmd_audio = [
                "ffmpeg",
                "-i",
                video_path,
                "-ar",
                str(self.target_sr),
                "-ac",
                "1",
                "-q:a",
                "0",
                audio_path,
                "-y",
            ]
            subprocess.run(cmd_audio, capture_output=True, check=False)

            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
            )

            return {
                "frames_dir": frames_dir,
                "audio_path": audio_path,
                "frame_count": len(frame_files),
                "success": len(frame_files) > 0 and os.path.exists(audio_path),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup_extraction(self, extraction_result: dict):
        """Clean up extracted files to save disk space."""
        try:
            if extraction_result.get("frames_dir") and os.path.exists(
                extraction_result["frames_dir"]
            ):
                import shutil

                shutil.rmtree(extraction_result["frames_dir"])
            if extraction_result.get("audio_path") and os.path.exists(
                extraction_result["audio_path"]
            ):
                os.remove(extraction_result["audio_path"])
        except:
            pass


if __name__ == "__main__":
    print("Testing SegmentedDataset...")

    dataset = SegmentedFakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=8,
        image_size=160,
        segment_duration=4.0,
        max_samples=100,
        ensure_balance=True,
    )

    print(f"\nLoaded {len(dataset)} segments")

    if len(dataset) > 0:
        vis, aud, lbl = dataset[0]
        print(
            f"Sample shape: Visual={vis.shape}, Audio={aud.shape}, Label={lbl.item():.0f}"
        )
