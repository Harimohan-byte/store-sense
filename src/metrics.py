"""StoreSense: events.csv + zones.yaml -> zone KPIs, z-scores, flags -> output/report.csv"""
import pandas as pd
from zones import load

cfg, polys, types = load()
ev = pd.read_csv("output/events.csv")
slow_cut = 0.5 * ev.speed.median()       # "slow" = under half the typical speed
ev["slow"] = ev.speed < slow_cut

ADVICE = {
    "HIGH DWELL": "Dwell above norm: check for blocked path, stockout or confusing layout.",
    "LOW DWELL": "Dwell below norm: visitors leave quickly; check display, lighting, staff.",
    "DEAD ZONE": "No visitors: consider relocating items or changing the path.",
    "OK": "Within expected range.",
}

rows = []
for zone in polys:
    prof = cfg["profiles"][types[zone]]
    g = ev[ev.activity == zone]
    n = g.case_id.nunique()
    if n == 0:
        rows.append(dict(zone=zone, type=types[zone], visitors=0, avg_dwell_s=0,
                         slow_share=0, z_score=None, flag="DEAD ZONE"))
        continue
    mean_d = g.dur.mean()
    z = (mean_d - prof["dwell_mean"]) / prof["dwell_std"]
    flag = "HIGH DWELL" if z > 2 else "LOW DWELL" if z < -2 else "OK"
    rows.append(dict(zone=zone, type=types[zone], visitors=n,
                     avg_dwell_s=round(mean_d, 1),
                     slow_share=round(float(g.slow.mean()), 2),
                     z_score=round(z, 2), flag=flag))

rep = pd.DataFrame(rows)
rep["advice"] = rep.flag.map(ADVICE)
rep["confidence"] = ["low (n<10)" if n < 10 else "ok" for n in rep.visitors]
rep.to_csv("output/report.csv", index=False)
print(f"Slow cut-off: {slow_cut:.0f} px/s (half the median visit speed)\n")
print(rep.drop(columns="advice").to_string(index=False))
print()
for r in rep.itertuples():
    print(f"{r.zone}: {r.advice}")
