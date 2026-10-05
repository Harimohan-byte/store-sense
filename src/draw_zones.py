"""Click polygons on the first video frame.
Usage: python src/draw_zones.py data/store.mp4 Entry:entrance Middle:occasional Exit:checkout
Keys (click the image window first): left-click = add point | n = finish zone | u = undo | q = quit"""
import sys
import cv2
import numpy as np
import yaml

if len(sys.argv) < 3:
    sys.exit("Usage: python src/draw_zones.py <video> Name:type [Name:type ...]")
src = sys.argv[1]
specs = [a.split(":") for a in sys.argv[2:]]

cap = cv2.VideoCapture(src)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit("Cannot read video")
cv2.imwrite("output/first_frame.png", frame)

H, W = frame.shape[:2]
scale = min(1.0, 900 / H)
done, pts = [], []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((int(x / scale), int(y / scale)))

cv2.namedWindow("zones")
cv2.setMouseCallback("zones", on_click)
idx = 0
while idx < len(specs):
    name, ztype = specs[idx]
    img = frame.copy()
    for n, _, p in done:
        cv2.polylines(img, [np.array(p)], True, (0, 200, 0), 4)
        cv2.putText(img, n, p[0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 4)
    for p in pts:
        cv2.circle(img, p, 10, (0, 0, 255), -1)
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts)], False, (0, 0, 255), 4)
    cv2.putText(img, f"Draw {name} ({ztype}): n=finish u=undo", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4)
    cv2.imshow("zones", cv2.resize(img, None, fx=scale, fy=scale))
    k = cv2.waitKey(30) & 0xFF
    if k == ord("n") and len(pts) >= 3:
        done.append((name, ztype, list(pts)))
        pts.clear()
        idx += 1
    elif k == ord("u") and pts:
        pts.pop()
    elif k == ord("q"):
        sys.exit("Aborted, nothing saved")
cv2.destroyAllWindows()

cfg = {
    "zones": [{"name": n, "type": t, "polygon": [list(p) for p in P]}
              for n, t, P in done],
    "profiles": {
        "entrance":   {"dwell_mean": 3,  "dwell_std": 2},
        "daily_need": {"dwell_mean": 8,  "dwell_std": 4},
        "occasional": {"dwell_mean": 15, "dwell_std": 8},
        "checkout":   {"dwell_mean": 20, "dwell_std": 10},
    },
}
with open("zones.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Saved zones.yaml with", len(done), "zones")
