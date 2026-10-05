"""Overlay zones and track points on the first frame; report coverage."""
import cv2
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point, Polygon

cfg = yaml.safe_load(open("zones.yaml"))
d = pd.read_csv("output/tracks.csv")
polys = {z["name"]: Polygon(z["polygon"]) for z in cfg["zones"]}

def zone_of(x, y):
    p = Point(x, y)
    for name, poly in polys.items():
        if poly.contains(p):
            return name
    return "outside"

d["zone"] = [zone_of(x, y) for x, y in zip(d.x, d.y)]
print((d.zone.value_counts(normalize=True) * 100).round(1).to_string())

img = cv2.imread("output/first_frame.png")
for z in cfg["zones"]:
    cv2.polylines(img, [np.array(z["polygon"])], True, (0, 200, 0), 4)
    cv2.putText(img, z["name"], tuple(z["polygon"][0]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 4)
for x, y, zn in zip(d.x[::5], d.y[::5], d.zone[::5]):
    col = (0, 0, 255) if zn == "outside" else (255, 160, 0)
    cv2.circle(img, (int(x), int(y)), 4, col, -1)
cv2.imwrite("output/zones_preview.png", img)
print("Saved output/zones_preview.png")
