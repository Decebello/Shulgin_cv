import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.transform import resize

alphabet_img = imread('alphabet_ext.png', as_gray=True)
alphabet_binary = alphabet_img < 0.5
alphabet_props = sorted(regionprops(label(alphabet_binary)), key=lambda r: r.bbox[1])
char_names = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '-', '/', 'P', 'D']
templates = {char_names[i]: prop.image for i, prop in enumerate(alphabet_props)}

symbols_img = imread('symbols.png')
symbols_gray = np.mean(symbols_img[..., :3], axis=2) if symbols_img.ndim >= 3 else symbols_img
symbols_props = regionprops(label(symbols_gray > 0))

freq_dict = {char: 0 for char in char_names}

for prop in symbols_props:
    best_match = None
    best_score = -1
    for char, template in templates.items():
        resized_sym = resize(prop.image, template.shape) > 0.5
        union = np.logical_or(resized_sym, template).sum()
        score = np.logical_and(resized_sym, template).sum() / union if union > 0 else 0
        if score > best_score:
            best_score = score
            best_match = char
    if best_match:
        freq_dict[best_match] += 1

print(freq_dict)