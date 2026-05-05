import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CSV Pfad SSL
# =========================
csv_path = "results/semi_supervised_training/run_20260504_140626_mean_teacher_split20_thr09/history.csv"  

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/plots_semi_supervised"
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

plt.plot(
    epochs,
    df["train_total_loss"],
    color="orange",
    linestyle="-",
    linewidth=2.5,
    label="Trainingsverlust"
)

plt.plot(
    epochs,
    df["val_teacher_loss"],
    color="orange",
    linestyle="--",
    linewidth=2.5,
    label="Validierungsverlust"
)

plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Trainings- und Validierungsverlust im Semi-Supervised Learning")

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "loss_ssl.png")
eps_path = os.path.join(output_dir, "loss_ssl.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Plot gespeichert:")
print(png_path)
print(eps_path)