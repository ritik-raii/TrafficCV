# src/utils.py
import os, yaml
from typing import Tuple

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def centroid_from_xyxy(xyxy):
    x1,y1,x2,y2 = xyxy
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def point_side(pt: Tuple[float,float], line: Tuple[Tuple[int,int], Tuple[int,int]]):
    (x1,y1),(x2,y2) = line
    x,y = pt
    return (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
