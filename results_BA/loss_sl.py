import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CSV Pfad
# =========================
csv_path = "results/supervised_training/run_20260501_190729_lr1e3_to_2e4_freeze7/history.csv"

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/plots_supervised"
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
    df["train_loss"],
    color="blue",
    linestyle="-",
    linewidth=2.5,
    label="Trainingsverlust"
)

plt.plot(
    epochs,
    df["val_loss"],
    color="blue",
    linestyle="--",
    linewidth=2.5,
    label="Validierungsverlust"
)

plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Trainings- und Validierungsverlust im Supervised Learning")

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "loss_supervised.png")
eps_path = os.path.join(output_dir, "loss_supervised.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Plot gespeichert:")
print(png_path)
print(eps_path)