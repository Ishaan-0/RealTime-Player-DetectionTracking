import cv2
import os
from detector.best_detector import YOLOv11Detector
#from trackers.deepsort_tracker import DeepSORTWrapper as DeepTracker
from trackers.bytetrack_tracker import ByteTrackWrapper as ByteTracker
from utils.video_utils import draw_tracks
import config

def main():
    detector = YOLOv11Detector(config.MODEL_PATH)
    tracker = ByteTracker()

    if not os.path.exists(config.VIDEO_PATH):
        print("Video file not found.")
        return

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % config.SKIP_FRAMES != 0:
            continue

        original = frame.copy()
        resized = cv2.resize(frame, config.RESIZE_DIM)

        yolo_results = detector.detect(resized)

        tracks = tracker.track(yolo_results, frame)
        if frame_id % config.DRAW_EVERY_N == 0:
            original = draw_tracks(original, tracks)

        out.write(original)
    
    cap.release()
    out.release()
    print("Tracking complete.")
    print(f"Output saved to: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    main()
