"""StoreSense: tracks.csv -> output/events.csv (single-camera event log)."""
import pandas as pd
from zones import load, zone_of

GAP = 3.0         # s: a time gap larger than this starts a new visit
MIN_VISIT = 0.5   # s: ignore shorter visits (edge clipping)

cfg, polys, types = load()
d = pd.read_csv("output/tracks.csv").sort_values(["track_id", "t"]).reset_index(drop=True)

# smooth jitter in the foot points
g = d.groupby("track_id")
for c in ("x", "y"):
    d[c + "m"] = g[c].transform(lambda s: s.rolling(5, center=True, min_periods=1).median())

d["z"] = [zone_of(x, y, polys) or "outside" for x, y in zip(d.xm, d.ym)]
g = d.groupby("track_id")
d["dt"] = g.t.diff()
d["step"] = ((g.xm.diff() ** 2 + g.ym.diff() ** 2) ** 0.5).fillna(0)
prev_z = g.z.shift()
d["new"] = (d.z != prev_z) | (d.dt > GAP) | d.dt.isna()
d.loc[d.new, "step"] = 0
d["visit"] = d.new.cumsum()

v = d.groupby("visit").agg(case_id=("track_id", "first"), activity=("z", "first"),
                           start=("t", "min"), end=("t", "max"), path=("step", "sum"))
v["dur"] = v.end - v.start
v["speed"] = v.path / v.dur.clip(lower=0.1)          # pixels per second
v = v[(v.activity != "outside") & (v.dur >= MIN_VISIT)]
v = v.sort_values(["case_id", "start"])

# merge same-zone events separated by a tiny gap (polygon-edge flicker)
MERGE_GAP = 1.0
merged = []
for cid, g in v.groupby("case_id"):
    cur = None
    for r in g.itertuples():
        if cur and cur["activity"] == r.activity and r.start - cur["end"] <= MERGE_GAP:
            cur["end"] = r.end
            cur["path"] += r.path
        else:
            if cur:
                merged.append(cur)
            cur = dict(case_id=cid, activity=r.activity, start=r.start,
                       end=r.end, path=r.path)
    if cur:
        merged.append(cur)
v = pd.DataFrame(merged)
v["dur"] = v.end - v.start
v["speed"] = v.path / v.dur.clip(lower=0.1)
v[["case_id", "activity", "start", "end", "dur", "speed"]].round(2).to_csv(
    "output/events.csv", index=False)

print(f"Event log: {len(v)} events, {v.case_id.nunique()} cases")
print(v[["case_id", "activity", "start", "end", "dur"]].round(1).to_string(index=False))

v["next"] = v.groupby("case_id").activity.shift(-1)
print("\nDirectly-follows (zone -> next zone):")
print(v.dropna(subset=["next"]).groupby(["activity", "next"]).size().to_string())
print("\nJourney variants:")
print(v.groupby("case_id").activity.apply(lambda s: " > ".join(s)).value_counts().to_string())
