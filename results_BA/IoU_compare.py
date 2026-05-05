import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CSV Pfade
# =========================
csv_sl = "results/supervised_training/run_20260501_190729_lr1e3_to_2e4_freeze7/history.csv"
csv_ssl = "results/semi_supervised_training/run_20260504_140626_mean_teacher_split20_thr09/history.csv"  # anpassen

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/comparison_IoU"
os.makedirs(output_dir, exist_ok=True)

# =========================
# CSV laden
# =========================
df_sl = pd.read_csv(csv_sl)
df_ssl = pd.read_csv(csv_ssl)

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.titleweight": "normal",
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

plt.plot(
    df_sl["epoch"],
    df_sl["val_iou"],
    color="blue",
    linestyle="-",
    linewidth=2.5,
    label="Supervised Learning"
)

plt.plot(
    df_ssl["epoch"],
    df_ssl["val_teacher_iou"],
    color="orange",
    linestyle="-",
    linewidth=2.5,
    label="Semi-Supervised Learning"
)

plt.xlabel("Epoche")
plt.ylabel("IoU")
plt.title("Validierungs-IoU von Supervised und Semi-Supervised Learning")

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "val_iou_sl_vs_ssl.png")
eps_path = os.path.join(output_dir, "val_iou_sl_vs_ssl.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Plot gespeichert:")
print(png_path)
print(eps_path)