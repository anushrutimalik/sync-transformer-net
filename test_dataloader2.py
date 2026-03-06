import torch
from fakeavceleb_dataset import FakeAVCelebDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def custom_collate(batch):
    vis_list = []
    aud_list = []
    lbl_list = []
    
    for i, (vis, aud, lbl) in enumerate(batch):
        vis_list.append(vis)
        aud_list.append(aud)
        lbl_list.append(lbl)
        
    vis_shape = vis_list[0].shape
    for i, vis in enumerate(vis_list):
        if vis.shape != vis_shape:
            print(f"VIS SHAPE MISMATCH in batch element {i}! Expected {vis_shape}, got {vis.shape}")
            
    aud_shape = aud_list[0].shape
    for i, aud in enumerate(aud_list):
        if aud.shape != aud_shape:
            print(f"AUD SHAPE MISMATCH in batch element {i}! Expected {aud_shape}, got {aud.shape}")
            
    return torch.stack(vis_list), torch.stack(aud_list), torch.stack(lbl_list)

def main():
    dataset = FakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=16,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate
    )
    
    print("Testing dataloader...")
    try:
        for batch_idx, (vis, aud, lbl) in enumerate(tqdm(dataloader)):
            pass
    except Exception as e:
        print(f"Exception caught in loop: {e}")

if __name__ == "__main__":
    main()
