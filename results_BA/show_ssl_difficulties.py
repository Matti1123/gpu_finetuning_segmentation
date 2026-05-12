import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CSV Pfad
# =========================
csv_path = "results/semi_supervised_training/run_20260424_125932_mean_teacher_split20_thr07/history.csv"

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/plots_ssl"
os.makedirs(output_dir, exist_ok=True)

# =========================
# CSV laden
# =========================
df = pd.read_csv(csv_path)

epochs = df["epoch"]

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

# Train IoU
plt.plot(
    epochs,
    df["train_iou"],
    color="orange",
    linestyle="-",
    linewidth=2.5,
    label="Train IoU"
)

# Validation Teacher IoU
plt.plot(
    epochs,
    df["val_teacher_iou"],
    color="orange",
    linestyle="--",
    linewidth=2.5,
    label="Validation Teacher IoU"
)

plt.xlabel("Epoche")
plt.ylabel("IoU")

plt.title("Trainings- und Validierungs-IoU im SSL-Ansatz")

plt.legend(loc="upper left")

plt.grid(True)
plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "iou_ssl.png")
eps_path = os.path.join(output_dir, "iou_ssl.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Plot gespeichert:")
print(png_path)
print(eps_path)