"""Write output/zone_frame.png from the video, using the frame index stored in zones.yaml."""
import sys
import cv2
import yaml

src = sys.argv[1] if len(sys.argv) > 1 else "data/store2.mp4"
cfg = yaml.safe_load(open("zones.yaml"))
cap = cv2.VideoCapture(src)
cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.get("frame", 0))
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit("Cannot read the reference frame")
cv2.imwrite("output/zone_frame.png", frame)
print("Saved output/zone_frame.png")
