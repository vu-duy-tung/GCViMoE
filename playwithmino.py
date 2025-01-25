import os
import numpy as np

val_info = np.loadtxt("imagenet/val/val_annotations.txt", dtype='str')

labels = set()
img_idx_to_label = {}

for sample in val_info:
    labels.add(sample[1])
    img_idx_to_label[sample[0]] = sample[1]

for img_idx in os.listdir("imagenet/val/images"):
    label = img_idx_to_label[img_idx]
    old_path = os.path.join("imagenet/val/images", img_idx)
    new_path = os.path.join("imagenet/val", label)
    os.makedirs(new_path, exist_ok=True)
    os.system(f"mv {old_path} {new_path}")

import code; code.interact(local=locals())