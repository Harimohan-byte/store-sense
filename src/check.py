"""Quick tracking sanity check."""
import pandas as pd

d = pd.read_csv("output/tracks.csv")
g = d.groupby("track_id").t.agg(["min", "max"])
g["span"] = g["max"] - g["min"]
print("Unique track IDs:", d.track_id.nunique())
print("Tracks shorter than 2 s (likely noise or ID switches):",
      int((g.span < 2).sum()))
print("Median track length (s):", round(g.span.median(), 1))
print("Longest track (s):", round(g.span.max(), 1))
