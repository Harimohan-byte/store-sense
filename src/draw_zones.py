"""Pick a frame, then click zone polygons on it.
Usage: python src/draw_zones.py data/store.mp4 Entry:entrance Middle:occasional Exit:checkout [--frame N]
Step 1 (frame picker): drag the slider | ENTER or SPACE = use this frame | g = guide on/off | q = quit
Step 2 (drawing): left-click = add point | n = finish zone | u = undo | g = guide on/off | q = quit
Click the image window once so it has focus before pressing keys."""
import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("zones", nargs="+", help="Name:type")
ap.add_argument("--frame", type=int, default=None, help="skip the slider, use this frame")
ap.add_argument("--height", type=int, default=None, help="window height in pixels, e.g. 600")
args = ap.parse_args()
specs = [z.split(":") for z in args.zones]

cap = cv2.VideoCapture(args.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total < 1:
    sys.exit("Cannot read video")

def get_frame(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    return fr if ok else None

first = get_frame(0)
H, W = first.shape[:2]
def screen_height():
    try:
        import ctypes
        return ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        return 800

win_h = args.height or int(screen_height() * 0.70)
scale = min(1.0, win_h / H)
print(f"Window height: {int(H * scale)} px (scale {scale:.2f})")

guide = []
if os.path.exists("output/tracks.csv"):
    t = pd.read_csv("output/tracks.csv")
    guide = list(zip(t.x.astype(int)[::3], t.y.astype(int)[::3]))

def paint_guide(img, show):
    if show:
        for p in guide:
            cv2.circle(img, p, 3, (255, 160, 0), -1)

def show_img(img):
    cv2.imshow("zones", cv2.resize(img, None, fx=scale, fy=scale))

show = True
frame, chosen = first, 0

# ---- Step 1: pick the frame ----
cv2.namedWindow("zones")
if args.frame is None:
    cv2.createTrackbar("frame", "zones", 0, total - 1, lambda v: None)
    last = -1
    while True:
        pos = cv2.getTrackbarPos("frame", "zones")
        if pos != last:
            fr = get_frame(pos)
            if fr is not None:
                frame = fr
            last = pos
        img = frame.copy()
        paint_guide(img, show)
        cv2.putText(img, f"Frame {pos}: ENTER/SPACE = use, g = guide", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4)
        show_img(img)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):
            chosen = pos
            break
        elif k == ord("g"):
            show = not show
        elif k == ord("q"):
            sys.exit("Aborted, nothing saved")
else:
    chosen = args.frame
    frame = get_frame(chosen)
    if frame is None:
        sys.exit("Cannot read that frame")
cv2.destroyWindow("zones")

# ---- Step 2: draw zones ----
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
    paint_guide(img, show)
    for n, _, p in done:
        cv2.polylines(img, [np.array(p)], True, (0, 200, 0), 4)
        cv2.putText(img, n, p[0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 4)
    for p in pts:
        cv2.circle(img, p, 10, (0, 0, 255), -1)
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts)], False, (0, 0, 255), 4)
    cv2.putText(img, f"Draw {name} ({ztype}): n=finish u=undo g=guide", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4)
    show_img(img)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("n") and len(pts) >= 3:
        done.append((name, ztype, list(pts)))
        pts.clear()
        idx += 1
    elif k == ord("u") and pts:
        pts.pop()
    elif k == ord("g"):
        show = not show
    elif k == ord("q"):
        sys.exit("Aborted, nothing saved")
cv2.destroyAllWindows()
cap.release()

cv2.imwrite("output/zone_frame.png", frame)
cfg = {
    "frame": chosen,
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
print(f"Saved zones.yaml with {len(done)} zones on frame {chosen}")
