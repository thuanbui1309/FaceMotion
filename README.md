# FaceMotion

---
## Requirements

- Python = 3.10
- Install dependencies:

```bash
pip install opencv-python numpy pandas matplotlib facenet-pytorch ultralytics
```
or 
```bash
pip install -r requirements.txt
```
## Folder Structure
```text
FaceMotion/
├── main.py                      # Main script
├── models/
│   └── yolov8n-face.pt          # YOLOv8 face model
├── data/
│   └── demo.mp4                 # Input video
├── known_faces/
│   ├── Alice/
│   │   └── 1.jpg
│   └── Bob/
│       └── 1.jpg
├── face_samples/                # Auto-generated cropped face samples
├── runs/
│   ├── motion_activity.csv      # Movement log
│   ├── motion_plot.png          # Line chart of motion
│   └── activity_index.png       # Bar chart of average activity
```
## Face Registration
Prepare the `known_faces/` directory with subfolders named after each person:
```text
known_faces/
├── Alice/
│   ├── 1.jpg
│   └── 2.jpg
├── Bob/
│   ├── 1.jpg
│   └── 2.jpg
```

- Folder name = person's name.
- Use clear, frontal, unobstructed face images.
- 1–5 images per person is recommended.
## How to Run

Execute:

```bash
python main.py
```

Output will be stored in folder `runs`, which includes:
   - CSV log: `motion_activity.csv`
   - Line chart: `motion_plot.png`
   - Activity bar chart: `activity_index.png`