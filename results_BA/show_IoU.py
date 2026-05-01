import pandas as pd

csv_path = "results_BA/best_worst_testset/iou_scores.csv"

df = pd.read_csv(csv_path)

mean_iou = df["iou"].mean()
std_iou = df["iou"].std()

print(f"Mean IoU: {mean_iou:.4f}")
print(f"Std IoU:  {std_iou:.4f}")
print(f"Min IoU:  {df['iou'].min():.4f}")
print(f"Max IoU:  {df['iou'].max():.4f}")