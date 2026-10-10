"""Create the README images in docs/img from the latest pipeline outputs."""
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image

os.makedirs("docs/img", exist_ok=True)

for src, dst in [("output/heatmap.png", "docs/img/heatmap.png"),
                 ("output/zones_preview.png", "docs/img/zones.png")]:
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("copied", dst)

color = lambda i: (int((i * 37) % 200 + 55), int((i * 91) % 200 + 55), int((i * 53) % 200 + 55))

# all trajectories on the reference frame
base = cv2.imread("output/zone_frame.png")
base = cv2.addWeighted(base, 0.6, np.zeros_like(base), 0.4, 0)
d = pd.read_csv("output/tracks.csv")
for tid, g in d.groupby("track_id"):
    pts = g.sort_values("t")[["x", "y"]].astype(int).values.reshape(-1, 1, 2)
    cv2.polylines(base, [pts], False, color(tid), 3)
    x0, y0 = int(pts[0][0][0]), int(pts[0][0][1])
    cv2.circle(base, (x0, y0), 8, color(tid), -1)
    cv2.putText(base, f"P{tid}", (x0 + 10, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color(tid), 2)
cv2.imwrite("docs/img/trajectories.png", base)
print("saved docs/img/trajectories.png")

# annotated frame and short GIF from the rendered video
cap = cv2.VideoCapture("output/annotated_clean.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 13
cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
ok, fr = cap.read()
if ok:
    cv2.imwrite("docs/img/annotated.png", fr)
    print("saved docs/img/annotated.png")

cap.set(cv2.CAP_PROP_POS_FRAMES, n // 4)
step = max(1, int(fps / 6))
frames, i = [], 0
while len(frames) < 60:
    ok, fr = cap.read()
    if not ok:
        break
    if i % step == 0:
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (640, int(h * 640 / w)))
        frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    i += 1
cap.release()
if frames:
    frames[0].save("docs/img/demo.gif", save_all=True, append_images=frames[1:],
                   duration=170, loop=0, optimize=True)
    print("saved docs/img/demo.gif with", len(frames), "frames")
