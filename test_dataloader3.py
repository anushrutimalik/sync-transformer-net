import torch
from fakeavceleb_dataset import FakeAVCelebDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    dataset = FakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=16,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    print("Testing dataloader with pin_memory=True ...")
    try:
        for batch_idx, (vis, aud, lbl) in enumerate(tqdm(dataloader)):
            pass
    except Exception as e:
        print(f"Exception caught in loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
