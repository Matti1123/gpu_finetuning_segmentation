import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CSV Pfad
# =========================
csv_path = "results/classifier_exp_7/train_log.csv"

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/plots_classification"
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
    df["train_acc"],
    color="darkgreen",
    linestyle="-",
    linewidth=2.5,
    label="Trainingsgenauigkeit"
)

plt.plot(
    epochs,
    df["val_acc"],
    color="darkgreen",
    linestyle="--",
    linewidth=2.5,
    label="Validierungsgenauigkeit"
)

plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.title("Trainings- und Validierungsgenauigkeit des Klassifikationsmodells")

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "classification_accuracy.png")
eps_path = os.path.join(output_dir, "classification_accuracy.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Plot gespeichert:")
print(png_path)
print(eps_path)