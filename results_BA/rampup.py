import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Konfiguration
# =========================

RAMPUP_EPOCHS = 12
UNSUP_WEIGHT_MAX = 0.15
TOTAL_EPOCHS = 25

OUTPUT_DIR = "results_BA/plots_ssl"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Ramp-up Funktion
# =========================

epochs = np.arange(0, TOTAL_EPOCHS + 1)

rampup = np.exp(
    -5 * (
        1 - np.minimum(epochs, RAMPUP_EPOCHS) / RAMPUP_EPOCHS
    ) ** 2
)

# Nach Ramp-up konstant halten
rampup[epochs > RAMPUP_EPOCHS] = 1.0

lambda_u = UNSUP_WEIGHT_MAX * rampup


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

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    lambda_u,
    color="orange",
    linewidth=2.5
)

plt.xlabel("Epoche")
plt.ylabel("Gewichtung des Consistency Loss")

plt.title(
    "Verlauf der Gewichtung des\n"
    "Consistency Loss"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.xlim(0, TOTAL_EPOCHS)
plt.ylim(0, 0.155)

plt.tight_layout()


# =========================
# Speichern
# =========================

png_path = os.path.join(
    OUTPUT_DIR,
    "consistency_weight_orange.png"
)

eps_path = os.path.join(
    OUTPUT_DIR,
    "consistency_weight_orange.eps"
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