import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

num_frames = 100

frame0 = np.load('out/h_0.npy')
labeled0, n_obj = ndimage.label(frame0)
print(f"Количество объектов: {n_obj}")

trajectories = {i: [] for i in range(n_obj)}

for f in range(num_frames):
    frame = np.load(f'out/h_{f}.npy')
    labeled, n = ndimage.label(frame)
    centers = ndimage.center_of_mass(frame, labeled, range(1, n + 1))

    if f == 0:
        sorted_centers = sorted(centers, key=lambda c: c[1])
        for i, c in enumerate(sorted_centers):
            trajectories[i].append((f, c[0], c[1]))
        prev_centers = sorted_centers
    else:
        new_prev = [None] * n_obj
        for obj_id in range(n_obj):
            prev = prev_centers[obj_id]
            best_dist = float('inf')
            best_c = None
            for c in centers:
                d = (c[0] - prev[0]) ** 2 + (c[1] - prev[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_c = c
            trajectories[obj_id].append((f, best_c[0], best_c[1]))
            new_prev[obj_id] = best_c
        prev_centers = new_prev

fig, ax = plt.subplots(figsize=(8, 8))
colors = ['red', 'blue', 'green']

for obj_id in range(n_obj):
    traj = trajectories[obj_id]
    xs = [t[2] for t in traj]  
    ys = [t[1] for t in traj]  
    ax.plot(xs, ys, color=colors[obj_id], linewidth=1.5, label=f'Объект {obj_id + 1}')
    ax.scatter([xs[0]], [ys[0]], color=colors[obj_id], s=80, marker='o', zorder=5)
    ax.scatter([xs[-1]], [ys[-1]], color=colors[obj_id], s=80, marker='*', zorder=5)

ax.set_xlim(0, 600)
ax.set_ylim(600, 0)  
ax.set_xlabel('X (пиксели)')
ax.set_ylabel('Y (пиксели)')
ax.set_title('Траектории движения объектов (100 кадров)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trajectories.png', dpi=150)
plt.show()
print("График сохранён в trajectories.png")