"""Draw dropped raw tracks in red with the reason -> output/debug_dropped.mp4"""
import sys
import cv2
import pandas as pd

src = sys.argv[1]
raw = pd.read_csv("output/tracks_raw.csv")
drop = pd.read_csv("output/dropped.csv").set_index("raw_id").reason.to_dict()
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("output/debug_dropped.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
raw["f"] = (raw.t * fps).round().astype(int)
by_f = {f: g for f, g in raw.groupby("f")}
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    if i in by_f:
        for r in by_f[i].itertuples():
            if r.track_id in drop:
                cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{r.track_id} {drop[r.track_id]}",
                            (r.x1, max(r.y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    vw.write(frame)
cap.release()
vw.release()
print("Done -> output/debug_dropped.mp4")
