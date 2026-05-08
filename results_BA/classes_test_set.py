import matplotlib.pyplot as plt
import os

# =========================
# Daten
# =========================
classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
counts = [22, 526, 9, 8, 20, 0, 1]

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/plots_classification"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.titleweight": "normal",
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

bars = plt.bar(
    classes,
    counts,
    color="darkgreen",
    edgecolor="black",
    linewidth=1.0
)

# Werte über Balken anzeigen
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(count),
        ha='center',
        va='bottom',
        fontsize=11
    )

# Logarithmische Skalierung für bessere Sichtbarkeit
plt.yscale("log")

plt.xlabel("Vorhergesagte Läsionsklasse")
plt.ylabel("Anzahl Bilder")

plt.title(
    "Klassenverteilung der Testbilder"
)

plt.grid(True, axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "class_distribution.png")
eps_path = os.path.join(output_dir, "class_distribution.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Plot gespeichert:")
print(png_path)
print(eps_path)