import torch

try:
    ckpt = torch.load('syncweld_epoch_8.pth', map_location='cpu')
    m = ckpt.get('val_metrics', {})
    
    # 1. Main Metrics Table
    print("\n| Metric | Value (%) |")
    print("|--------|-----------|")
    print(f"| Accuracy | {m.get('acc', 0)*100:.2f}% |")
    print(f"| Precision | {m.get('prec', 0)*100:.2f}% |")
    print(f"| Recall (Sensitivity) | {m.get('rec', 0)*100:.2f}% |")
    print(f"| F1 Score | {m.get('f1', 0)*100:.2f}% |")
    print(f"| ROC AUC | {m.get('auc', 0)*100:.2f}% |")
except Exception as e:
    print(f"Error reading checkpoint: {e}")
