import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path

output_dir = Path(__file__).parent / "recognition_results"
output_dir.mkdir(exist_ok=True)

def get_feature_vector(region_obj):
    img = region_obj.image.astype(float)
    h, w = img.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=float)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = img

    resized = resize(square, (20, 20), order=1, anti_aliasing=False)
    return resized.flatten()

def predict_symbol(target_features, reference_dict):
    best_match = "?"
    min_dist = float('inf')

    for sym, ref_features in reference_dict.items():
        dist = np.linalg.norm(ref_features - target_features)
        if dist < min_dist:
            min_dist = dist
            best_match = sym

    return best_match

template_image = imread("alphabet-small.png")[:, :, :3].sum(axis=2)
binary_template = template_image != 765.0
labeled_template = label(binary_template)
template_props = regionprops(labeled_template)

template_props_sorted = sorted(template_props, key=lambda p: p.bbox[1])

known_classes = ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/"]
reference_data = {}
for prop, symbol in zip(template_props_sorted, known_classes):
    reference_data[symbol] = get_feature_vector(prop)

target_image = imread("alphabet.png")[:, :, :3]
binary_target = target_image.mean(axis=2) > 0
labeled_target = label(binary_target)
print(f"Кол-во найденных объектов: {np.max(labeled_target)}")

target_props = regionprops(labeled_target)

stats = {}
plt.figure(figsize=(5, 7))
for idx, prop in enumerate(target_props):
    predicted = predict_symbol(get_feature_vector(prop), reference_data)

    stats[predicted] = stats.get(predicted, 0) + 1

    plt.clf()
    plt.title(f"класс: '{predicted}'")
    plt.imshow(prop.image, cmap='gray')
    plt.savefig(output_dir / f"char_{prop.label:03d}.png")

print("\nСтатистика распознавания:", stats)
errors = stats.get("?", 0)
accuracy = (1 - errors / len(target_props)) * 100
print(f"Процент распознавания: {accuracy:.2f}%")

plt.imshow(binary_target, cmap='gray')
plt.title("Изображение для распознавания")
plt.show()