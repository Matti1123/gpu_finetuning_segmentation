# Hautläsions-Segmentierung mit Deep Learning (ISIC Datensatz)

## Projektbeschreibung

Dieses Projekt untersucht die automatische Segmentierung von Hautläsionen mithilfe von Deep-Learning-Methoden.  
Als Datengrundlage wird der öffentlich verfügbare **ISIC Datensatz** verwendet, der Dermatoskopie-Bilder und zugehörige Segmentierungsmasken enthält.

Ziel des Projekts ist es, verschiedene **neuronale Netzwerkarchitekturen** sowie **Hyperparameter-Konfigurationen** im Rahmen von **Supervised Learning** zu vergleichen und deren Einfluss auf die Segmentierungsqualität zu analysieren.

---

## Ziele

- Implementierung einer Trainingspipeline für Bildsegmentierung  
- Vergleich verschiedener Modelle (z. B. **DeepLabV3**, **U-Net**)  
- Analyse des Einflusses von Hyperparametern  
- Evaluation der Modelle anhand geeigneter Metriken (z. B. **IoU – Intersection over Union**)

---

## Datensatz

Verwendet wird der **ISIC Skin Lesion Segmentation Datensatz**.

Der Datensatz enthält:

- Dermatoskopie-Bilder von Hautläsionen  
- Manuell erstellte Segmentierungsmasken  

Diese Masken dienen als **Ground Truth** für das Training der Modelle.
