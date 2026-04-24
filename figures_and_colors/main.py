import cv2
import numpy as np
from collections import defaultdict
import sys
import os

def get_color_name(h, s, v):
    if s < 60:
        return "серый"
    if h < 10 or h >= 170:
        return "красный"
    elif h < 25:
        return "оранжевый"
    elif h < 45:
        return "жёлтый"
    elif h < 75:
        return "жёлто-зелёный"
    elif h < 100:
        return "зелёный"
    elif h < 120:
        return "голубой"
    elif h < 140:
        return "синий"
    elif h < 170:
        return "фиолетовый/розовый"
    return "неизвестный"

def classify_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return "круг" if circularity > 0.75 else "прямоугольник"

def get_contour_color(img, contour):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_bgr = cv2.mean(img, mask=mask)[:3]
    pixel = np.uint8([[list(mean_bgr)]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return get_color_name(int(hsv[0]), int(hsv[1]), int(hsv[2]))

def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось открыть изображение '{image_path}'")
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) >= 5]

    total = 0
    circle_colors = defaultdict(int)
    rect_colors = defaultdict(int)

    for contour in contours:
        shape = classify_shape(contour)
        if shape is None:
            continue
        color = get_contour_color(img, contour)
        total += 1
        if shape == "круг":
            circle_colors[color] += 1
        else:
            rect_colors[color] += 1

    print("=" * 50)
    print(f"  Общее количество фигур: {total}")
    print("=" * 50)

    total_circles = sum(circle_colors.values())
    total_rects = sum(rect_colors.values())

    print(f"\nКруги — всего: {total_circles}")
    for color, count in sorted(circle_colors.items(), key=lambda x: -x[1]):
        print(f"  {color}: {count}")

    print(f"\nПрямоугольники — всего: {total_rects}")
    for color, count in sorted(rect_colors.items(), key=lambda x: -x[1]):
        print(f"  {color}: {count}")

    print("\nОбщая статистика по оттенкам:")
    all_colors = defaultdict(int)
    for color, count in circle_colors.items():
        all_colors[color] += count
    for color, count in rect_colors.items():
        all_colors[color] += count
    for color, count in sorted(all_colors.items(), key=lambda x: -x[1]):
        print(f"  {color}: {count}")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "balls_and_rects.png")

    analyze_image(image_path)