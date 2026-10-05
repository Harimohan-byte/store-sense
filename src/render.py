"""StoreSense: draw cleaned tracks and trails -> output/annotated_clean.mp4"""
import sys
import cv2
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/store.mp4"
d = pd.read_csv("output/tracks.csv")
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("output/annotated_clean.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
d["f"] = (d.t * fps).round().astype(int)
by_f = {f: g for f, g in d.groupby("f")}
trails = {}
color = lambda i: ((i * 37) % 200 + 55, (i * 91) % 200 + 55, (i * 53) % 200 + 55)

i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    if i in by_f:
        for r in by_f[i].itertuples():
            trails.setdefault(r.track_id, []).append((int(r.x), int(r.y)))
    for tid, pts in trails.items():
        pts[:] = pts[-60:]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color(tid), 3)
    if i in by_f:
        for r in by_f[i].itertuples():
            cv2.circle(frame, (int(r.x), int(r.y)), 8, color(r.track_id), -1)
            cv2.putText(frame, f"P{r.track_id}", (int(r.x) + 10, int(r.y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color(r.track_id), 2)
    vw.write(frame)

cap.release(); vw.release()
print("Done -> output/annotated_clean.mp4")
