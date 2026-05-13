import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# Pfade
# =====================================================

sl_iou_csv = "results_BA/best_worst_testset/iou_scores.csv"
ssl_iou_csv = "results_BA/best_worst_testset_ssl/iou_scores.csv"

class_csv = "results_BA/Classification_results/test_class_predictions_0.75.csv"

# =====================================================
# Output
# =====================================================

output_dir = "results_BA/plots_classwise_segmentation"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# CSVs laden
# =====================================================

df_sl = pd.read_csv(sl_iou_csv)
df_ssl = pd.read_csv(ssl_iou_csv)
df_class = pd.read_csv(class_csv)

# =====================================================
# Dateinamen extrahieren
# =====================================================

df_sl["filename"] = df_sl["image"].apply(os.path.basename)
df_ssl["filename"] = df_ssl["image"].apply(os.path.basename)
df_class["filename"] = df_class["image_path"].apply(os.path.basename)

# =====================================================
# Nur sichere Vorhersagen
# =====================================================

df_class = df_class[df_class["above_0_75"] == True]

# =====================================================
# Merge
# =====================================================

df_sl = pd.merge(
    df_sl,
    df_class[["filename", "predicted_class"]],
    on="filename",
    how="inner"
)

df_ssl = pd.merge(
    df_ssl,
    df_class[["filename", "predicted_class"]],
    on="filename",
    how="inner"
)

# =====================================================
# Klassen auswählen
# =====================================================

valid_classes = ["NV", "MEL", "BKL", "BCC", "AKIEC"]

df_sl = df_sl[df_sl["predicted_class"].isin(valid_classes)]
df_ssl = df_ssl[df_ssl["predicted_class"].isin(valid_classes)]

# =====================================================
# Statistik berechnen
# =====================================================

sl_stats = df_sl.groupby("predicted_class")["iou"].mean()
ssl_stats = df_ssl.groupby("predicted_class")["iou"].mean()

classes = valid_classes

sl_mean = [sl_stats[c] if c in sl_stats.index else 0 for c in classes]
ssl_mean = [ssl_stats[c] if c in ssl_stats.index else 0 for c in classes]

# =====================================================
# Style für BA-Subplots
# =====================================================

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.titleweight": "normal",
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# =====================================================
# Plot-Funktion
# =====================================================

def save_classwise_plot(values, color, title, filename_base):
    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        classes,
        values,
        color=color,
        edgecolor="black",
        linewidth=1.2
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=17
        )

    plt.ylabel("Mean IoU")
    plt.xlabel("Vorhergesagte Läsionsklasse")
    plt.title(title, fontsize=24)

    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    png_path = os.path.join(output_dir, filename_base + ".png")
    eps_path = os.path.join(output_dir, filename_base + ".eps")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(eps_path, format="eps", bbox_inches="tight")

    plt.close()

    print(f"Gespeichert: {png_path}")
    print(f"Gespeichert: {eps_path}")

# =====================================================
# SL Plot
# =====================================================

save_classwise_plot(
    values=sl_mean,
    color="blue",
    title="Segmentierungsleistung nach Läsionsklasse (SL)",
    filename_base="classwise_iou_sl"
)

# =====================================================
# SSL Plot
# =====================================================

save_classwise_plot(
    values=ssl_mean,
    color="orange",
    title="Segmentierungsleistung nach Läsionsklasse (SSL)",
    filename_base="classwise_iou_ssl"
)

# =====================================================
# Terminal Output
# =====================================================

print("\n===== Mean IoU SL =====")
for c, v in zip(classes, sl_mean):
    print(f"{c}: {v:.4f}")

print("\n===== Mean IoU SSL =====")
for c, v in zip(classes, ssl_mean):
    print(f"{c}: {v:.4f}")

print("\nPlots gespeichert.")