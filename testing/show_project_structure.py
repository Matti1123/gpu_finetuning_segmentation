import os

def print_tree(startpath, prefix=""):
    files = os.listdir(startpath)
    files.sort()
    
    for i, name in enumerate(files):
        path = os.path.join(startpath, name)
        is_last = i == len(files) - 1
        
        connector = "`-- " if is_last else "|-- "
        print(prefix + connector + name)
        
        if os.path.isdir(path):
            extension = "    " if is_last else "|   "
            print_tree(path, prefix + extension)

# Pfad zu deinem Projektordner
print("bachelorarbeit_segmentierung/")
print_tree("/home/mci/gpu_finetuning_segmentation")