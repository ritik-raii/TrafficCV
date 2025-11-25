# src/counter.py
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Dict

from .utils import centroid_from_xyxy, point_side

@dataclass
class TrackState:
    last_centroid: Tuple[float,float] = None
    class_id: int = None
    counted: bool = False
    frames: int = 0

class Counter:
    def __init__(self, line: Tuple[Tuple[int,int], Tuple[int,int]], min_frames=2, min_bbox_area=400):
        self.line = line
        self.min_frames = min_frames
        self.min_bbox_area = min_bbox_area
        self.tracks: Dict[int, TrackState] = {}
        self.counts = defaultdict(int)

    def _bbox_area(self, xyxy):
        x1,y1,x2,y2 = xyxy
        return max(0, (x2-x1)) * max(0, (y2-y1))

    def process_detection(self, track_id: int, class_id: int, xyxy: Tuple[float,float,float,float], frame_no:int):
        if track_id is None:
            return
        cen = centroid_from_xyxy(xyxy)
        state = self.tracks.get(track_id, TrackState(last_centroid=None, class_id=class_id, counted=False, frames=0))
        state.frames += 1
        state.class_id = class_id
        # simple bbox size filter
        if self._bbox_area(xyxy) < self.min_bbox_area:
            state.last_centroid = cen
            self.tracks[track_id] = state
            return
        # crossing logic
        if state.last_centroid is not None and not state.counted:
            s1 = point_side(state.last_centroid, self.line)
            s2 = point_side(cen, self.line)
            if s1 * s2 < 0 and state.frames >= self.min_frames:
                # crossed
                self.counts[class_id] += 1
                state.counted = True
        state.last_centroid = cen
        self.tracks[track_id] = state

    def get_counts(self):
        return dict(self.counts)
