import matplotlib.pyplot as plt

def render_table(data, columns, title, filename, col_widths=None):
    # Adjust height based on number of rows
    fig, ax = plt.subplots(figsize=(10, len(data)*0.6 + 1.0))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0') # Seaborn default blue
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('#ffffff')
                
        # Make the first column text bold for contrast
        if col == 0 and row != 0:
            cell.set_text_props(weight='bold')
            
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# Table 1 Data
t1_cols = ["Metric", "Value (%)", "Description"]
t1_data = [
    ["Accuracy", "97.00%", "Overall percentage of correctly classified validation videos"],
    ["Precision", "100.00%", "No pristine (real) videos falsely accused as deepfakes"],
    ["Recall (Sensitivity)", "93.75%", "Percentage of actual deepfakes successfully caught"],
    ["F1-Score", "96.77%", "Harmonic mean proving excellent metric balance"],
    ["ROC AUC", "96.83%", "Outstanding discriminatory capability across thresholds"]
]
# Adjusted column widths to make text fit perfectly
render_table(t1_data, t1_cols, "Table 1: Overall Performance Metrics for SyncWeld-Net (Epoch 8)", "paper_table1_overall_metrics.png", col_widths=[0.2, 0.15, 0.65])


# Table 2 Data
t2_cols = ["Modality / Manipulation Type", "Sub-Accuracy (%)", "Validation Samples"]
t2_data = [
    ["RealVideo-RealAudio (Control)", "100.00%", "104"],
    ["FakeVideo-FakeAudio (Full Deepfake)", "97.73%", "44"],
    ["FakeVideo-RealAudio (Visual-only)", "97.92%", "48"],
    ["RealVideo-FakeAudio (Audio-only)", "0.00%", "4"]
]
render_table(t2_data, t2_cols, "Table 2: Accuracy Breakdown by Deepfake Modality", "paper_table2_modality_breakdown.png", col_widths=[0.45, 0.25, 0.25])
