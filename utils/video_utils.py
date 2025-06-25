import cv2

def draw_tracks(frame, tracks):
    for trk in tracks:
        # DeepSORT Track object
        if hasattr(trk, "is_confirmed") and callable(trk.is_confirmed):
            if not trk.is_confirmed():
                continue

            track_id = getattr(trk, "track_id", None)
            if hasattr(trk, "to_ltwh"):
                l, t, w, h = map(int, trk.to_ltwh())
                x2, y2 = l + w, t + h
            else:
                continue  # Skip if bbox is malformed

        # ByteTrack tuple
        elif isinstance(trk, tuple) and len(trk) == 2:
            bbox, track_id = trk
            x1, y1, x2, y2 = map(int, bbox)
            l, t = x1, y1

        else:
            continue  # Skip unknown object types

        cv2.rectangle(frame, (l, t), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame
