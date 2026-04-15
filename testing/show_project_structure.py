import os

path = "/home/mci/gpu_finetuning_segmentation"

print("bachelorarbeit_segmentierung/")
for item in os.listdir(path):
    print(f"|-- {item}")