import os
import csv
import time
import torch
from collections import Counter
from fakeavceleb_dataset import FakeAVCelebDataset
from torch.utils.data import DataLoader

def analyze_dataset():
    csv_path = "datasets/FakeAVCeleb_v1.2/meta_data.csv"
    base_path = "datasets/FakeAVCeleb_v1.2"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print("--- 1. Analyzing Metadata ---")
    total_entries = 0
    existing_files = 0
    categories = Counter()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_entries += 1
            rel_prefix = row.get('', '')
            if rel_prefix.startswith('FakeAVCeleb/'):
                rel_prefix = rel_prefix.replace('FakeAVCeleb/', '', 1)
                
            full_path = os.path.join(base_path, rel_prefix, row['path'])
            categories[row['type']] += 1
            
            if os.path.exists(full_path):
                existing_files += 1

    print(f"Total entries in CSV: {total_entries}")
    print(f"Total physically existing files: {existing_files}")
    print("\nCategory breakdown in CSV (Real vs Fake types):")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
        
    print("\n--- 2. Benchmarking Data Loading Speed ---")
    try:
        # Load a small dataset piece to benchmark
        dataset = FakeAVCelebDataset(
            metadata_csv_path=csv_path,
            base_path=base_path,
            num_frames=8, # from training config
            image_size=160, # from training config
        )
        # Take a subset of 32 samples for benchmarking
        benchmark_subset_size = min(32, len(dataset))
        dataset.metadata = dataset.metadata[:benchmark_subset_size]
        
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        print(f"Benchmarking with batch_size={batch_size}, num_workers={4}")
        start_time = time.time()
        
        batches_processed = 0
        samples_processed = 0
        for vis, aud, lbl in dataloader:
            # Simulate moving to GPU and pass
            vis = vis.cuda() if torch.cuda.is_available() else vis
            aud = aud.cuda() if torch.cuda.is_available() else aud
            lbl = lbl.cuda() if torch.cuda.is_available() else lbl
            batches_processed += 1
            samples_processed += vis.size(0)
            
        end_time = time.time()
        
        elapsed = end_time - start_time
        time_per_sample = elapsed / max(1, samples_processed)
        
        print(f"Processed {samples_processed} samples in {elapsed:.2f} seconds.")
        print(f"Average time per sample: {time_per_sample:.4f} seconds")
        
        print("\n--- 3. 20-Hour Training Projection ---")
        epochs = 10
        target_hours = 20
        target_seconds = target_hours * 3600
        seconds_per_epoch = target_seconds / epochs
        
        # Assume training pipeline (forward+backward) takes maybe 1.5x to 2x the dataloading time 
        # (This is conservative; actual compute might be faster or slower, but data loading is often bottleneck without pre-processing)
        estimated_total_time_per_sample = time_per_sample * 2.0 
        
        max_samples_per_epoch = int(seconds_per_epoch / estimated_total_time_per_sample)
        print(f"Target time per epoch: {seconds_per_epoch:.0f} seconds ({seconds_per_epoch/3600:.2f} hours)")
        print(f"Estimated total step time (load + train) per sample: {estimated_total_time_per_sample:.4f} seconds")
        print(f"-> Estimated MAXIMUM samples for {target_hours}hr training (10 epochs): {max_samples_per_epoch}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")

if __name__ == '__main__':
    analyze_dataset()
