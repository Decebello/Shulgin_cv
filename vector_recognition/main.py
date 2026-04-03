import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects, zoom

def get_feature_vector(roi):
    h, w = roi.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=float)
    square[(size - h) // 2:(size - h) // 2 + h, (size - w) // 2:(size - w) // 2 + w] = roi
    
    resized = zoom(square, 20.0 / size, order=1)
    out = np.zeros((20, 20), dtype=float)
    rh, rw = resized.shape
    out[:min(20, rh), :min(20, rw)] = resized[:min(20, rh), :min(20, rw)]
    
    return out.flatten()

def main():
    img_t = Image.open('alphabet-small.png').convert('RGBA')
    bg = Image.new('RGBA', img_t.size, (255, 255, 255, 255))
    temp_arr = np.array(Image.alpha_composite(bg, img_t).convert('L'))
    thresh_temp = (temp_arr < 128).astype(int)

    struct = np.ones((3, 3), dtype=int)
    labeled_temp, _ = label(thresh_temp, structure=struct)
    boxes = [(s[1].start, s[0].start, s) for s in find_objects(labeled_temp) if s]
    boxes.sort(key=lambda b: b[0])

    labels = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '-', '/']
    templates = {labels[i]: get_feature_vector(thresh_temp[b[2]]) for i, b in enumerate(boxes) if i < len(labels)}

    img_tgt = Image.open('alphabet.png').convert('RGB')
    thresh_tgt = (np.array(img_tgt.convert('L')) > 1).astype(int)

    labeled_tgt, _ = label(thresh_tgt, structure=struct)
    results = {l: 0 for l in labels}
    draw = ImageDraw.Draw(img_tgt)

    for s in find_objects(labeled_tgt):
        if not s or s[1].stop - s[1].start < 2 or s[0].stop - s[0].start < 2:
            continue
        
        vec = get_feature_vector(thresh_tgt[s])
        best_label = min(templates.keys(), key=lambda k: np.linalg.norm(vec - templates[k]))
        
        results[best_label] += 1
        draw.text((s[1].start, s[0].start - 10), best_label, fill=(255, 255, 255))

    for lbl, count in results.items():
        print(f"{lbl}: {count}")

    img_tgt.save('result.png')

if __name__ == '__main__':
    main()