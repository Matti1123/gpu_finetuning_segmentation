import os
import csv
import glob
from PIL import Image

import matplotlib.pyplot as plt


# =========================
# Konfiguration
# =========================

IMAGE_DIR = "data/classification_dataset/ISIC2018_Task3_Training_Input"
CSV_PATH = "data/classification_dataset/ISIC2018_Task3_Training_GroundTruth.csv"

OUTPUT_DIR = "results_BA/isic_class_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_NAME = "isic_classes_overview"

IMG_SIZE = 256


# =========================
# Klassen
# =========================

CLASS_ORDER = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

CLASS_TITLES = {
    "MEL": "MEL",
    "NV": "NV",
    "BCC": "BCC",
    "AKIEC": "AKIEC",
    "BKL": "BKL",
    "DF": "DF",
    "VASC": "VASC",
}


# =========================
# Bildsuche
# =========================

def find_image_path(image_id):
    possible_paths = [
        os.path.join(IMAGE_DIR, image_id + ".jpg"),
        os.path.join(IMAGE_DIR, image_id + ".png"),
        os.path.join(IMAGE_DIR, image_id + ".jpeg"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    matches = glob.glob(os.path.join(IMAGE_DIR, image_id + ".*"))

    if len(matches) > 0:
        return matches[0]

    return None


# =========================
# Beispielbilder laden
# =========================

def load_class_examples():
    examples = {}

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = row["image"]

            for class_name in CLASS_ORDER:
                value = row[class_name]

                if value in ["1", "1.0"] and class_name not in examples:

                    image_path = find_image_path(image_id)

                    if image_path is not None:
                        examples[class_name] = image_path

                        print(
                            f"{class_name}: "
                            f"{os.path.basename(image_path)}"
                        )

            if len(examples) == len(CLASS_ORDER):
                break

    missing = [
        c for c in CLASS_ORDER
        if c not in examples
    ]

    if len(missing) > 0:
        raise FileNotFoundError(
            f"Keine Beispielbilder gefunden für: {missing}"
        )

    return examples


# =========================
# Plot 4-3 Layout
# =========================

def save_class_overview(examples):

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7)
    )

    fig.patch.set_facecolor("white")

    axes = axes.flatten()

    # 4 oben, 3 unten
    positions = {
        "MEL": 0,
        "NV": 1,
        "BCC": 2,
        "AKIEC": 3,
        "BKL": 4,
        "DF": 5,
        "VASC": 6,
    }

    # Alle Achsen deaktivieren
    for ax in axes:
        ax.axis("off")

    for class_name in CLASS_ORDER:

        image_path = examples[class_name]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        ax = axes[positions[class_name]]

        ax.imshow(image)

        ax.set_title(
            CLASS_TITLES[class_name],
            fontsize=18,
            pad=8,
        )

        ax.axis("off")

    plt.tight_layout(
        pad=0.8,
        h_pad=1.0,
        w_pad=0.8,
    )

    png_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_NAME + ".png"
    )

    eps_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_NAME + ".eps"
    )

    # PNG
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    # EPS
    plt.savefig(
        eps_path,
        format="eps",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Fertig.")
    print(f"PNG: {png_path}")
    print(f"EPS: {eps_path}")


# =========================
# Main
# =========================

def main():

    examples = load_class_examples()

    save_class_overview(examples)


if __name__ == "__main__":
    main()