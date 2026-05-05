import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CSV Pfade
# =========================
csv_sl = "results_BA/best_worst_testset/iou_scores.csv"
csv_ssl = "results_BA/best_worst_testset_ssl/iou_scores.csv"

# =========================
# Output-Verzeichnis
# =========================
output_dir = "results_BA/Boxplots_original"
os.makedirs(output_dir, exist_ok=True)

# =========================
# CSV laden
# =========================
df_sl = pd.read_csv(csv_sl)
df_ssl = pd.read_csv(csv_ssl)

# =========================
# IoU extrahieren
# =========================
iou_sl = df_sl["iou"]
iou_ssl = df_ssl["iou"]

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.titleweight": "normal",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
})

# =========================
# Boxplot
# =========================
plt.figure(figsize=(8, 6))

box = plt.boxplot(
    [iou_sl, iou_ssl],
    patch_artist=True,
    labels=["SL", "SSL"],
    widths=0.5
)

# Farben
colors = ["blue", "orange"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# Median hervorheben
for median in box["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

# Whisker und Caps
for whisker in box["whiskers"]:
    whisker.set_linewidth(1.5)

for cap in box["caps"]:
    cap.set_linewidth(1.5)

plt.ylabel("IoU")
plt.title("Verteilung der IoU auf dem Testdatensatz")

plt.grid(True)
plt.tight_layout()

# =========================
# Speichern
# =========================
png_path = os.path.join(output_dir, "iou_boxplot_sl_vs_ssl.png")
eps_path = os.path.join(output_dir, "iou_boxplot_sl_vs_ssl.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print("Boxplot gespeichert:")
print(png_path)
print(eps_path)