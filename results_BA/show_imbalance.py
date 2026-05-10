import os
import csv
import matplotlib.pyplot as plt


# =========================
# Pfade
# =========================

CSV_PATH = (
    "data/classification_dataset/"
    "ISIC2018_Task3_Training_GroundTruth.csv"
)

OUTPUT_DIR = "results_BA/inbalance_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Klassen
# =========================

classes = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

counts = {
    c: 0
    for c in classes
}


# =========================
# CSV einlesen
# =========================

with open(CSV_PATH, "r", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        for class_name in classes:

            value = row[class_name]

            if value in ["1", "1.0"]:
                counts[class_name] += 1


# =========================
# Daten vorbereiten
# =========================

count_values = [
    counts[c]
    for c in classes
]

print("Klassenverteilung:")
for c in classes:
    print(f"{c}: {counts[c]}")


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
    count_values,
    color="darkgreen",
    edgecolor="black",
    linewidth=1.0
)


# =========================
# Werte über Balken
# =========================

for bar, count in zip(bars, count_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(count),
        ha="center",
        va="bottom",
        fontsize=11
    )


# =========================
# Labels
# =========================

plt.xlabel("Läsionsklasse")
plt.ylabel("Anzahl Bilder")

plt.title(
    "Klassenverteilung"
)

plt.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()


# =========================
# Speichern
# =========================

png_path = os.path.join(
    OUTPUT_DIR,
    "class_distribution_isic2018.png"
)

eps_path = os.path.join(
    OUTPUT_DIR,
    "class_distribution_isic2018.eps"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    eps_path,
    format="eps",
    bbox_inches="tight"
)

plt.close()


# =========================
# Ausgabe
# =========================

print("Plot gespeichert:")
print(png_path)
print(eps_path)