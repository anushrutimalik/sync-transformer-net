import os
import sys
import torch
import cv2
import numpy as np
import argparse
try:
    import torchaudio
except ImportError:
    print("WARNING: torchaudio not installed. Audio loading will fail.")

from models.syncweld import SyncWeldNet
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def extract_frames(video_path, num_frames=8, image_size=224):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        fallback = np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)
        return torch.from_numpy(fallback).permute(0, 3, 1, 2).float() / 255.0

    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        else:
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
    
    cap.release()
    frames_array = np.stack(frames)
    frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0
    return frames_tensor

def extract_audio(video_path, target_sample_rate=16000):
    """Extract audio directly from the video file."""
    fallback_audio = torch.zeros(target_sample_rate * 5)
    try:
        try:
            waveform, sample_rate = torchaudio.load(video_path)
        except Exception:
            import av
            container = av.open(video_path)
            try:
                audio_stream = container.streams.audio[0]
            except IndexError:
                container.close()
                return fallback_audio
            
            sample_rate = audio_stream.codec_context.sample_rate
            samples = []
            for frame in container.decode(audio_stream):
                samples.append(frame.to_ndarray())
            container.close()
            
            if len(samples) > 0:
                waveform_np = np.concatenate(samples, axis=-1)
                waveform = torch.from_numpy(waveform_np).float()
            else:
                return fallback_audio
        
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        max_length = target_sample_rate * 5 
        if waveform.shape[0] > max_length:
            waveform = waveform[:max_length]
        else:
            padding = max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform
        
    except Exception as e:
        print(f"Failed to load audio for {video_path}: {e}")
        return fallback_audio

def predict_video(video_path, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load configuration
    config = {
        'model': {
            'image-size': 160, # Reduced from 224 to massively boost batch speed
            'patch-size': 1,
            'num-classes': 1,
            'num-patches': 100, # 10x10 patches for 160 resolution
            'dim': 512,
            'num-frames': 8, # Cut completely in half to double execution speed
            'max-identities': 2,
            'depth': 9,
            'dim-head': 64,
            'heads': 8,
            'channels': 112, # Output channels of efficientnet block 7
            'attn-dropout': 0.1,
            'ff-dropout': 0.1,
            'shift-tokens': False,
            'enable-size-emb': False, # Avoid tracking sizes for generic training
            'enable-pos-emb': True,
            'enable-identity-attention': False,
            'efficient-net-block': 7
        }
    }
    
    # 2. Instantiate Model
    print("Initializing SyncWeldNet...")
    model = SyncWeldNet(timesformer_config=config, audio_model_name='facebook/wav2vec2-large-xlsr-53', num_classes=1)
    
    # 3. Load Weights
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        sys.exit(1)
        
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle DataParallel prefix if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 4. Extract data
    print(f"Processing video: {video_path}")
    frames = extract_frames(video_path, num_frames=config['model']['num-frames'], image_size=config['model']['image-size'])
    audio = extract_audio(video_path)
    
    frames = frames.unsqueeze(0).to(device) # Add batch dimension -> [1, Frames, C, H, W]
    audio = audio.unsqueeze(0).to(device)   # Add batch dimension -> [1, Audio_Length]
    
    # 5. Predict
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, _, _ = model(frames, audio)
            probability = torch.sigmoid(logits).item()
            
    print("-" * 50)
    print(f"Prediction Result for {os.path.basename(video_path)}:")
    if probability > 0.5:
        print(f"  Result: FAKE")
        print(f"  Confidence: {probability * 100:.2f}%")
    else:
        print(f"  Result: REAL (Pristine)")
        print(f"  Confidence: {(1.0 - probability) * 100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference of SyncWeldNet on a single video.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--weights", type=str, default="syncweld_epoch_8.pth", help="Path to the trained model weights.")
    
    args = parser.parse_args()
    predict_video(args.video_path, args.weights)
