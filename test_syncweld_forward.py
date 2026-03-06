import torch
import torch.nn as nn
from models.syncweld import SyncWeldNet
from train_syncweld import ContrastiveDissonanceLoss
import sys
import traceback

def run_dry_test():
    print("Initiating SyncWeld-Net Dry Test...")
    
    # 1. Setup minimal config for TimeSformer
    config = {
        'model': {
            'image-size': 224,
            'patch-size': 1,
            'num-classes': 1,
            'num-patches': 49,
            'dim': 512,
            'num-frames': 8, # tiny batch
            'max-identities': 2,
            'depth': 1,      # very shallow for memory and speed
            'dim-head': 64,
            'heads': 4,
            'channels': 32,  # simulating output from EfficientNet patch extraction
            'attn-dropout': 0.1,
            'ff-dropout': 0.1,
            'shift-tokens': False,
            'enable-size-emb': False, # skipping extra embeddings for a simple test
            'enable-pos-emb': False,
            'enable-identity-attention': False
        }
    }
    
    try:
        # Initialize the network
        print("Constructing Model...")
        # Since Wav2Vec2 is massive, this will download the weights if not cached
        model = SyncWeldNet(timesformer_config=config, num_classes=1)
        model.eval()
        print("Model Construction Successful.")
        
        # 2. Create Dummy Data
        # Batch=2
        # Visual tensor (simulating embedded sequence from visual encoder) 
        # Actually SizeInvariantTimeSformer expects input shape specific to its configuration 
        # which usually handles patches. Let's create dummy input: [Batch, Frames, Patches, Channels] or [Batch, SeqLen, Channels]. 
        # TimeSformer in this repo might expect [Batch, Frames * Patches, Channels]. We'll mock 8 frames, 1 patch, 32 channels.
        
        batch_size = 2
        frames = config['model']['num-frames']
        channels = config['model']['channels'] # EfficientNet extracted features space
        h = 7 # 7x7 patches = 49
        w = 7
        
        # MINTIME SizeInvariantTimeSformer expects pre-embedded patch maps
        # x.shape = b, f, c, h, w => [2, 8, 32, 7, 7]
        visual_dummy = torch.randn(batch_size, frames, channels, h, w)
        
        # Audio Dummy: 16kHz audio, 1 second = 16000 samples
        audio_dummy = torch.randn(batch_size, 16000)
        
        # MINTIME requires explicit position and identity masks due to its specialized Multi-Identity tracking
        positions = torch.arange(0, frames * h * w).unsqueeze(0).repeat(batch_size, 1)
        mask_dummy = torch.ones(batch_size, frames, dtype=torch.bool)
        id_mask_dummy = torch.ones(batch_size, frames, frames, dtype=torch.bool)
        
        labels_dummy = torch.tensor([[1], [0]], dtype=torch.float32)

        print("Created Dummy Tensors. Running Forward Pass...")
        
        # 3. Forward Pass
        # Disabling gradient just to verify dimension shapes
        with torch.no_grad():
            logits, latents, features = model(
                visual_x=visual_dummy, 
                audio_waveforms=audio_dummy,
                mask=mask_dummy,
                identities_mask=id_mask_dummy,
                positions=positions
            )
            print(f"Forward Pass Successful!")
            print(f"-> Logits Shape: {logits.shape}")
            print(f"-> Audio Latents Shape: {latents.shape}")
            print(f"-> Visual Features Shape: {features.shape}")
            
        # 4. Loss Test
        criterion = ContrastiveDissonanceLoss(alpha=0.5)
        loss, cls, cd = criterion(logits, labels_dummy, latents, features)
        print(f"Loss Calculation Successful! Total: {loss.item():.4f} | CLS: {cls.item():.4f} | CD: {cd.item():.4f}")
        
    except Exception as e:
        print("Dry test failed with error:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_dry_test()
