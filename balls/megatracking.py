import cv2
import numpy as np
from pathlib import Path
import json
import random 

colors = ["red", "yellow", "orange"]
random_colors = random.shuffle(colors)

save_path = Path(__file__).parent
config_path = save_path / "config.json"

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

position = [0, 0]
clicked = False

def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        global position 
        global clicked
        position = [x, y]
        clicked = True

cv2.setMouseCallback("Image", on_click)
cam = cv2.VideoCapture(0)

targets = []

if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        if "targets" in js:
            for t in js["targets"]:
                targets.append({
                    "lower": np.array(t["lower"], dtype="u1"),
                    "upper": np.array(t["upper"], dtype="u1"),
                    "positions": []
                })

d = 6.36 #cm

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
        
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
        
    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]
        lower = np.clip(color * 0.8, 0, 255).astype("u1")
        upper = np.clip(color * 1.2, 0, 255).astype("u1")
        
        if len(targets) >= 3:
            targets.pop(0)
            
        targets.append({
            "lower": lower,
            "upper": upper,
            "positions": []
        })

    combined_mask = None 

    for target in targets:
        inr = cv2.inRange(hsv, target["lower"], target["upper"])
        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x,y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
                target["positions"].append((x, y))
                if len(target["positions"]) > 20:
                    target["positions"].pop(0)
                    
        for i, pos in enumerate(target["positions"][:-1]):
            color_val = int(100 + 155 / len(target["positions"]) * i)
            cv2.circle(frame, pos, i*2, (0, 0, color_val), -1)
            
    if combined_mask is not None:
        cv2.imshow("Mask", combined_mask)
                    
    cv2.imshow("Image", frame)
    
cam.release()
cv2.destroyAllWindows()

save_data = {"targets": []}
for t in targets:
    save_data["targets"].append({
        "lower": t["lower"].tolist(),
        "upper": t["upper"].tolist()
    })

with config_path.open("w") as f:
    json.dump(save_data, f, indent=4)