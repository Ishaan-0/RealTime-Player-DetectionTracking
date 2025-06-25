# ⚽ Player Tracking from Soccer Footage

This project uses a **fine-tuned YOLOv11 model** to detect soccer players and a tracking algorithm (**DeepSORT** or **ByteTrack**) to assign persistent IDs—even if a player leaves and re-enters the frame.

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
### Install Dependencies 
```bash
pip install -r requirements.txt
```
### Project Structure 
```bash
.
├── config.py                  # Configuration settings
├── main.py                    # Entry point
├── detector/
│   └── best_detector.py       # YOLOv11 detection logic
├── trackers/
│   ├── base_tracker.py        # Abstract base class
│   ├── bytetrack_tracker.py   # ByteTrack implementation
│   └── deepsort_tracker.py    # DeepSORT implementation
├── utils/
│   └── video_utils.py         # Drawing logic
├── Best.pt                    # Fine-tuned YOLOv11 model (not included in the repo)
├── 15sec_input_720p.mp4       # Sample input video (not included in the repo)
├── requirements.txt           # Python dependencies
└── README.md
```

### Control how tracking and detection behave using the config.py file
``` bash
MODEL_PATH = "Best.pt"
VIDEO_PATH = "15sec_input_720p.mp4"
OUTPUT_PATH = "output_video.mp4"

RESIZE_DIM = (640, 360)         # Resize input frame for faster inference
SKIP_FRAMES = 2                 # Skip every N frames to speed up processing
DRAW_EVERY_N = 2                # Draw every N frames
CLASSES_TO_DETECT = [0]        # Class 0 = Player
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
```

