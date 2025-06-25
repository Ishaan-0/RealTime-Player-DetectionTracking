from trackers.base_tracker import BaseTracker
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSORTWrapper(BaseTracker):
    def __init__(self):
        try:
            self.tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2)
        except Exception as e:
            print(f"Error initializing DeepSORT: {e}")
            raise

    def track(self, yolo_results, frame=None):
        detections = []

        for box, conf, cls in zip(yolo_results.boxes.xyxy, 
                                  yolo_results.boxes.conf, 
                                  yolo_results.boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            if w<=0 or h<=0:
                continue
            bbox = [x1, y1, w, h]
            detections.append((bbox, float(conf), int(cls)))

        # DeepSORT returns a list of track objects
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
