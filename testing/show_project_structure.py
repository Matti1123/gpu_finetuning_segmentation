import os

def print_tree(startpath, max_depth=2, prefix="", level=0):
    if level > max_depth:
        return
    
    files = sorted(os.listdir(startpath))
    
    for i, name in enumerate(files):
        # Skip unnötige Sachen
        if name in ["__pycache__", ".git", "venv", ".venv"] or name.endswith(".pyc"):
            continue
        
        path = os.path.join(startpath, name)
        is_last = i == len(files) - 1
        
        connector = "`-- " if is_last else "|-- "
        print(prefix + connector + name)
        
        if os.path.isdir(path):
            extension = "    " if is_last else "|   "
            print_tree(path, max_depth, prefix + extension, level + 1)

# Aufruf
print("bachelorarbeit_segmentierung/")
print_tree("/home/mci/gpu_finetuning_segmentation", max_depth=2)