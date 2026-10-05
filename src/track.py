"""StoreSense step 1: video -> output/tracks_raw.csv + output/annotated_raw.mp4"""
import csv
import sys
import cv2
from ultralytics import YOLO

src = sys.argv[1] if len(sys.argv) > 1 else "data/store.mp4"
weights = sys.argv[2] if len(sys.argv) > 2 else "yolov8s.pt"
ENHANCE = True            # CLAHE contrast boost for weak lighting

model = YOLO(weights)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

cap = cv2.VideoCapture(src)
if not cap.isOpened():
    sys.exit(f"Cannot open video: {src}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}x{H} @ {fps:.1f} fps, {total} frames, model {weights}")

vw = cv2.VideoWriter("output/annotated_raw.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
f = open("output/tracks_raw.csv", "w", newline="")
out = csv.writer(f)
out.writerow(["track_id", "t", "x", "y", "x1", "y1", "x2", "y2", "conf"])

i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    inp = enhance(frame) if ENHANCE else frame
    res = model.track(inp, persist=True, classes=[0], conf=0.25, imgsz=960,
                      tracker="configs/botsort_store.yaml", verbose=False)[0]
    if res.boxes.id is not None:
        for box, tid, c in zip(res.boxes.xyxy, res.boxes.id, res.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            tid = int(tid)
            out.writerow([tid, round(i / fps, 2), (x1 + x2) // 2, y2,
                          x1, y1, x2, y2, round(float(c), 2)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{tid} {float(c):.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    vw.write(frame)
    if i % 100 == 0:
        print(f"frame {i}/{total}")

cap.release(); vw.release(); f.close()
print("Done -> output/tracks_raw.csv, output/annotated_raw.mp4")
