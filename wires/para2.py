import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

image=np.load("C:/Users/shulg_yq3nszk/OneDrive/Desktop/Lektsia/wires4.npy")
struct=np.ones((3,1))
process=binary_opening(image,struct)

labeled_image=label(image)
labeled_process=label(process)

print(f"Original: {np.max(labeled_image)}")
print(f"Processed: {np.max(labeled_process)}")

for number in range(1, np.max(labeled_image)+1):
    wire=(labeled_image==number)
    wire_parts=(labeled_process[wire])
    unique_parts=np.unique(wire_parts[wire_parts>0])
    num_parts=len(unique_parts)
    print(f"провод {number},порван на {num_parts} части")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(process)
plt.show()