"""Zone helpers: load zones.yaml and map a point to a zone."""
import yaml
from shapely.geometry import Point, Polygon


def load(path="zones.yaml"):
    cfg = yaml.safe_load(open(path))
    polys = {z["name"]: Polygon(z["polygon"]) for z in cfg["zones"]}
    types = {z["name"]: z["type"] for z in cfg["zones"]}
    return cfg, polys, types


def zone_of(x, y, polys):
    p = Point(x, y)
    for name, poly in polys.items():
        if poly.covers(p):
            return name
    return None
