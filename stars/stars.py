import numpy as np
from skimage.measure import label, regionprops

stars = np.load("stars.npy")
labeled = label(stars > 0)
regions = regionprops(labeled)

count = 0
for r in regions:
    if r.area / r.bbox_area < 0.6:
        count += 1

print(count)