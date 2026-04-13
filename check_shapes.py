from fakeavceleb_dataset import FakeAVCelebDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    dataset = FakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=16,
    )
    
    print("Dataset size:", len(dataset))
    
    vis_shape = None
    aud_shape = None
    
    for i in tqdm(range(len(dataset))):
        vi
        s, aud, lbl = dataset[i]
        
        if vis_shape is None:
            vis_shape = vis.shape
            aud_shape = aud.shape
        else:
            if vis.shape != vis_shape:
                print(f"Shape mismatch at index {i}: expected vis {vis_shape}, got {vis.shape}")
                break
            if aud.shape != aud_shape:
                print(f"Shape mismatch at index {i}: expected aud {aud_shape}, got {aud.shape}")
                break

if __name__ == "__main__":
    main()
