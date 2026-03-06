"""
FakeAVCeleb Dataset Loader for SyncWeld-Net

This script acts as the data handler for the FakeAVCeleb dataset, extracting and pairing:
1. Video Frames (Visual Tokens) - Cropped faces, structurally similar to MINTIME processing.
2. Raw Audio Waveforms (Audio Tokens) - Handled natively by Torchaudio to feed the Wav2Vec2 Engine.
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
try:
    import torchaudio
except ImportError:
    print("WARNING: torchaudio not installed. Audio loading will fail during execution.")

from sklearn.model_selection import train_test_split

class FakeAVCelebDataset(Dataset):
    def __init__(self, metadata_csv_path, base_path, num_frames=16, image_size=224, target_sample_rate=16000, transform=None, split='all', val_ratio=0.2, max_samples=None):
        """
        Args:
            metadata_csv_path (str): Path to the fakeavceleb meta_data.csv file.
            base_path (str): Root directory of the FakeAVCeleb dataset (where category folders exist).
            num_frames (int): Number of frames to extract per video.
            image_size (int): Resolution to resize the face crops.
            target_sample_rate (int): Required sample rate for Wav2Vec2 model (usually 16kHz).
            transform: Optional torchvision transforms for data augmentation on frames.
            split: 'train', 'val', or 'all'.
            val_ratio: Proportion of dataset to use for validation.
            max_samples: Maximum number of samples to keep for this split to restrict training time.
        """
        import csv
        all_metadata = []
        with open(metadata_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Resolve partial path
                rel_prefix = row.get('', '')
                if rel_prefix.startswith('FakeAVCeleb/'):
                    rel_prefix = rel_prefix.replace('FakeAVCeleb/', '', 1)
                    
                video_file_name = row['path']
                full_video_path = os.path.join(base_path, rel_prefix, video_file_name)
                
                # Omit rows where the physical file doesn't exist to prevent IO blocking
                if os.path.exists(full_video_path):
                    all_metadata.append(row)
                
        # Group rows by label for balancing
        real_metadata = []
        fake_metadata = []
        
        for row in all_metadata:
            category = row['type']
            label = 0 if category == 'RealVideo-RealAudio' else 1
            if label == 0:
                real_metadata.append(row)
            else:
                fake_metadata.append(row)
                
        # Undersample the fake class to perfectly match the real class size
        import random
        random.seed(42) # Ensuring reproducibility in the random sample
        min_class_size = min(len(real_metadata), len(fake_metadata))
        
        real_metadata = random.sample(real_metadata, min_class_size)
        fake_metadata = random.sample(fake_metadata, min_class_size)
        
        # Re-combine the balanced dataset
        balanced_metadata = real_metadata + fake_metadata
        random.shuffle(balanced_metadata)
        
        # Split logic
        if split in ['train', 'val']:
            # Use deterministic seed (42) to ensure train/val splits are consistent across runs
            train_data, val_data = train_test_split(balanced_metadata, test_size=val_ratio, random_state=42)
            self.metadata = train_data if split == 'train' else val_data
        else:
            self.metadata = balanced_metadata
            
        if max_samples is not None:
            self.metadata = self.metadata[:max_samples]
            
        self.base_path = base_path
        self.num_frames = num_frames
        self.image_size = image_size
        self.target_sample_rate = target_sample_rate
        self.transform = transform

    def _extract_frames(self, video_path):
        """Extract evenly spaced frames from a video."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If the video is shorter than the requested frames, we just take what we can and pad later.
        if frame_count == 0:
            fallback = np.zeros((self.num_frames, self.image_size, self.image_size, 3), dtype=np.uint8)
            return torch.from_numpy(fallback).permute(0, 3, 1, 2).float() / 255.0

        # Calculate indices for evenly spaced frames
        frame_indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to standard model size (224x224)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            else:
                # Fallback zero-frame if read fails
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        cap.release()
        
        # Convert to numpy array [Frames, H, W, Channels]
        frames_array = np.stack(frames)
        
        # Convert to PyTorch format [Frames, Channels, H, W]
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0
        return frames_tensor

    def _extract_audio(self, video_path):
        """Extract audio directly from the video file or corresponding .wav file."""
        # Note: torchaudio can often read audio directly from mp4 files depending on the backend.
        # However, for FakeAVCeleb, there are often separate .wav files, or we rely on FFmpeg.
        
        # Ensure fallback audio matches the standardized sequence length of 5 seconds (80000 samples for 16kHz)
        fallback_audio = torch.zeros(self.target_sample_rate * 5)
        try:
            try:
                waveform, sample_rate = torchaudio.load(video_path)
            except Exception as e:
                # Fallback to PyAV if torchaudio fails (common on Windows without FFmpeg)
                import av
                import numpy as np
                container = av.open(video_path)
                try:
                    audio_stream = container.streams.audio[0]
                except IndexError:
                    container.close()
                    raise ValueError("No audio track found in video")
                
                sample_rate = audio_stream.codec_context.sample_rate
                samples = []
                for frame in container.decode(audio_stream):
                    samples.append(frame.to_ndarray())
                container.close()
                
                if len(samples) > 0:
                    waveform_np = np.concatenate(samples, axis=-1)
                    waveform = torch.from_numpy(waveform_np).float()
                else:
                    raise ValueError("No audio frames extracted via PyAV")
            
            # Resample if necessary (Wav2Vec2 demands 16kHz)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
                
            # If stereo, convert to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Squeeze channel dim: [1, Audio_Length] -> [Audio_Length]
            waveform = waveform.squeeze(0)
            
            # Truncate or pad to a standardized length (e.g., 5 seconds = 80000 samples)
            max_length = self.target_sample_rate * 5 
            if waveform.shape[0] > max_length:
                waveform = waveform[:max_length]
            else:
                padding = max_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            return waveform
            
        except Exception as e:
            # Handle cases where video has no audio track
            print(f"Failed to load audio for {video_path}: {e}")
            return fallback_audio

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        
        # The CSV has an empty column name "" at the end that contains the relative path prefix
        # We strip the root 'FakeAVCeleb/' and join it with the base_path
        rel_prefix = row.get('', '')
        if rel_prefix.startswith('FakeAVCeleb/'):
            rel_prefix = rel_prefix.replace('FakeAVCeleb/', '', 1)
            
        video_file_name = row['path']
        full_video_path = os.path.join(self.base_path, rel_prefix, video_file_name)
        
        # 0 for Real, 1 for Fake
        category = row['type']
        label = 0 if category == 'RealVideo-RealAudio' else 1
        
        # 1. Processing Visually (Frames)
        visual_tensor = self._extract_frames(full_video_path)
        if self.transform:
            visual_tensor = torch.stack([self.transform(frame) for frame in visual_tensor])
            
        # 2. Processing Audio (1D Waveform)
        audio_tensor = self._extract_audio(full_video_path)
        
        return visual_tensor.clone(), audio_tensor.clone(), torch.tensor(label, dtype=torch.float32)

# Example Usage
if __name__ == "__main__":
    dataset = FakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2"
    )
    print(f"Dataset Loaded. Total samples: {len(dataset)}")
    vis, aud, lbl = dataset[0]
    print(f"Sample 0 - Vis Shape: {vis.shape}, Aud Shape: {aud.shape}, Label: {lbl}")

