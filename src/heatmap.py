"""StoreSense: tracks.csv -> output/heatmap.png (foot-traffic heat map over the zone frame)."""
import cv2
import numpy as np
import pandas as pd
import yaml

img = cv2.imread("output/zone_frame.png")
H, W = img.shape[:2]
d = pd.read_csv("output/tracks.csv")

heat = np.zeros((H, W), np.float32)
for x, y in zip(d.x, d.y):
    if 0 <= x < W and 0 <= y < H:
        heat[int(y), int(x)] += 1
heat = cv2.GaussianBlur(heat, (0, 0), 35)
if heat.max() > 0:
    heat = heat / heat.max()

color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
blend = cv2.addWeighted(img, 0.5, color, 0.5, 0)
out = np.where((heat > 0.05)[..., None], blend, img).astype(np.uint8)

cfg = yaml.safe_load(open("zones.yaml"))
for z in cfg["zones"]:
    poly = np.array(z["polygon"], dtype=np.int32)
    cv2.polylines(out, [poly], True, (255, 255, 255), 3)
    cv2.putText(out, z["name"], tuple(int(v) for v in poly[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
cv2.imwrite("output/heatmap.png", out)
print("Saved output/heatmap.png")
