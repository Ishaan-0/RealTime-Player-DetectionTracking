from trackers.base_tracker import BaseTracker
from supervision import ByteTrack, Detections

class ByteTrackWrapper(BaseTracker):
    def __init__(self):
        try: 
            self.tracker = ByteTrack()
            print("ByteTrack initialized successfully.")
        except Exception as e:
            print(f"Error initializing ByteTrack: {e}")
            raise

    def track(self, yolo_results, frame=None):
        detections = Detections.from_ultralytics(yolo_results)
        tracks = self.tracker.update_with_detections(detections)
        
        return tracks
