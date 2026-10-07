"""StoreSense step 1b: tracks_raw.csv -> tracks.csv (junk removed, fragments stitched)."""
import pandas as pd

RAW, OUT = "output/tracks_raw.csv", "output/tracks.csv"
STATIC_EXTENT = 60     # px: a track moving less than this is "static"
STATIC_CONF = 0.0
CONTAIN = 0.8          # box overlap share that counts as "inside another box"
CONTAIN_FRAC = 0.6     # share of a track's frames spent inside a larger box -> carried item
MAX_GAP = 3.0          # s: longest occlusion gap we will stitch across
D0, V_MAX = 120, 200   # px, px/s: stitch if dist <= D0 + V_MAX * gap
MIN_SPAN = 1.5         # s: shortest track kept after stitching

d = pd.read_csv(RAW).sort_values(["track_id", "t"])
n_raw = d.track_id.nunique()
dropped = {}

# 1) carried items (backpack): small box mostly inside a larger person box
d["area"] = (d.x2 - d.x1) * (d.y2 - d.y1)
inside = {}
for _, g in d.groupby("t"):
    r = g.to_dict("records")
    for a in r:
        for b in r:
            if a["track_id"] == b["track_id"] or b["area"] <= a["area"]:
                continue
            iw = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
            ih = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
            if iw > 0 and ih > 0 and iw * ih / a["area"] >= CONTAIN:
                inside[a["track_id"]] = inside.get(a["track_id"], 0) + 1
                break
cnt = d.groupby("track_id").size()
for tid, k in inside.items():
    if k / cnt[tid] >= CONTAIN_FRAC:
        dropped[tid] = "carried item"

# 2) static low-confidence objects (clothes lying on the floor)
for tid, g in d.groupby("track_id"):
    ext = max(g.x.max() - g.x.min(), g.y.max() - g.y.min())
    if tid not in dropped and ext < STATIC_EXTENT and g.conf.mean() < STATIC_CONF:
        dropped[tid] = "static object"

d = d[~d.track_id.isin(dropped)]

# 3) stitch fragments: link end of track A to start of track B across a short gap
tr = {tid: g.reset_index(drop=True) for tid, g in d.groupby("track_id")}
order = sorted(tr, key=lambda k: tr[k].t.iloc[0])
root = {k: k for k in tr}
has_next = set()
for b in order:
    tb = tr[b].t.iloc[0]
    xb, yb = tr[b].x.iloc[0], tr[b].y.iloc[0]
    best, best_d = None, 1e9
    for a in order:
        if a == b or a in has_next:
            continue
        gap = tb - tr[a].t.iloc[-1]
        if not (0 < gap <= MAX_GAP):
            continue
        dist = ((tr[a].x.iloc[-1] - xb) ** 2 + (tr[a].y.iloc[-1] - yb) ** 2) ** 0.5
        if dist <= D0 + V_MAX * gap and dist < best_d:
            best, best_d = a, dist
    if best is not None:
        has_next.add(best)
        root[b] = root[best]
d["track_id"] = d.track_id.map(root)

# 3b) staff: present most of the clip and barely moves -> excluded from customer KPIs
STAFF_FRAC, STAFF_EXTENT = 0.8, 300
clip_len = pd.read_csv(RAW).t.max()
span = d.groupby("track_id").t.agg(lambda s: s.max() - s.min())
rng = d.groupby("track_id").agg(xmin=("x", "min"), xmax=("x", "max"),
                                ymin=("y", "min"), ymax=("y", "max"))
ext = pd.concat([rng.xmax - rng.xmin, rng.ymax - rng.ymin], axis=1).max(axis=1)
staff = span.index[(span >= STAFF_FRAC * clip_len) & (ext < STAFF_EXTENT)]
d[d.track_id.isin(staff)][["track_id", "t", "x", "y"]].to_csv("output/tracks_staff.csv", index=False)
for tid in staff:
    dropped[tid] = "staff"
d = d[~d.track_id.isin(staff)]

# 4) drop short tracks after stitching
span = d.groupby("track_id").t.agg(lambda s: s.max() - s.min())
short = span[span < MIN_SPAN].index
for tid in short:
    dropped[tid] = "too short"
d = d[~d.track_id.isin(short)]

pd.Series(dropped, name="reason").rename_axis("raw_id").to_csv("output/dropped.csv")
d[["track_id", "t", "x", "y"]].to_csv(OUT, index=False)
print(f"Raw IDs: {n_raw} -> clean tracks: {d.track_id.nunique()}")
print("Dropped:", pd.Series(dropped).value_counts().to_dict() if dropped else "none")
print("Details:", dropped)
