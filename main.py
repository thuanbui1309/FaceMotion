import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.patches as mpatches
import shutil

# Param
MODEL = "models/yolov8n-face.pt"
VIDEO = "data/demo.mp4"
OUTPUT_SRC = "runs"
FACE_SAMPLE = "face_samples"
KNOWN_FACES = "known_faces"
DIST_THRESHOLD = 50
MAX_HISTORY = 30

# Init models
model = YOLO(MODEL)
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=True, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Video loop
prev_faces = {}
next_id = 0
track_history = {}
motion_log = {}
frame_idx = 0
os.makedirs(FACE_SAMPLE, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    curr_centers = []
    boxes = []

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        curr_centers.append((cx, cy))
        boxes.append((x1, y1, x2, y2))

    new_faces = {}
    matched_prev_ids = set()

    for center, box in zip(curr_centers, boxes):
        cx, cy = center
        matched_id = None
        min_dist = DIST_THRESHOLD

        for pid, (px, py) in prev_faces.items():
            dist = np.linalg.norm([cx - px, cy - py])
            if dist < min_dist and pid not in matched_prev_ids:
                matched_id = pid
                min_dist = dist

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        matched_prev_ids.add(matched_id)
        new_faces[matched_id] = (cx, cy)

        if matched_id not in track_history:
            track_history[matched_id] = deque(maxlen=MAX_HISTORY)

        track_history[matched_id].append((cx, cy))

        # Crop face and save
        x1, y1, x2, y2 = box
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0 and frame_idx % 50 == 0:
            save_dir = f"{FACE_SAMPLE}/id_{matched_id}"
            os.makedirs(save_dir, exist_ok=True)
            count = len(os.listdir(save_dir))
            cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_crop)

        # Calculate motion
        if matched_id in prev_faces:
            px, py = prev_faces[matched_id]
            dx, dy = cx - px, cy - py
            movement = np.sqrt(dx**2 + dy**2)
        else:
            movement = 0.0

        if matched_id not in motion_log:
            motion_log[matched_id] = []
        motion_log[matched_id].append((frame_idx, movement))

        # Draw box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {matched_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw trajectory
        pts = track_history[matched_id]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)

    prev_faces = new_faces.copy()
    frame_idx += 1

    cv2.imshow("FaceMotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Encode known_faces
def encode_known_faces(reference_dir):
    encodings = []
    names = []
    for person_name in os.listdir(reference_dir):
        person_dir = os.path.join(reference_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for file in os.listdir(person_dir):
            path = os.path.join(person_dir, file)
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is not None:
                face = face.unsqueeze(0)
                emb = resnet(face).detach().numpy().flatten()
                encodings.append(emb)
                names.append(person_name)
    return encodings, names

known_encodings, known_names = encode_known_faces(KNOWN_FACES)

# Match faces
id_to_name = {}
for id_folder in os.listdir(FACE_SAMPLE):
    id_path = os.path.join(FACE_SAMPLE, id_folder)
    if not os.path.isdir(id_path):
        continue

    matched = False
    for file in os.listdir(id_path):
        path = os.path.join(id_path, file)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is not None:
            face = face.unsqueeze(0)
            emb = resnet(face).detach().numpy().flatten()
            sims = [np.dot(emb, ref) for ref in known_encodings]
            if sims and max(sims) > 0.5:
                id_to_name[int(id_folder.split('_')[1])] = known_names[np.argmax(sims)]
                matched = True
                break
    if not matched:
        id_to_name[int(id_folder.split('_')[1])] = id_folder

# Store activity
records = []
for id_, log in motion_log.items():
    for frame_idx, movement in log:
        records.append((frame_idx, id_, movement))

df = pd.DataFrame(records, columns=["frame", "id", "movement"])
df["name"] = df["id"].map(id_to_name)
df.to_csv(f"{OUTPUT_SRC}/motion_activity.csv", index=False)

# Plot movement over time
plt.figure(figsize=(12, 6))
for name in df["name"].unique():
    df_pid = df[df["name"] == name].sort_values("frame")
    plt.plot(df_pid["frame"], df_pid["movement"].rolling(5, min_periods=1).mean(), label=name)

df_avg = df[df["frame"] >= 4]
if not df_avg.empty:
    avg_movement = df_avg["movement"].mean()
    plt.axhline(y=avg_movement, color='red', linestyle='--', label="Avg Active Level")

plt.title("Movement Activity Over Time")
plt.xlabel("Frame")
plt.ylabel("Movement Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_SRC}/motion_plot.png")

# Plot Activity Index
activity_df = df.groupby("name")["movement"].mean().reset_index()
activity_df = activity_df.sort_values("movement", ascending=False)

# Calculate threshold
mu = activity_df["movement"].mean()
sigma = activity_df["movement"].std()

# Match color with activeness
colors = []
for val in activity_df["movement"]:
    if val > mu + sigma:
        colors.append("red")
    elif val < mu - sigma:
        colors.append("gray")
    else:
        colors.append("skyblue")

# Draw activity chart
plt.figure(figsize=(10, 5))
bars = plt.bar(activity_df["name"], activity_df["movement"], color=colors)
plt.title("Activity Index per Student")
plt.xlabel("Name")
plt.ylabel("Activity Index")
plt.grid(axis="y")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha="center", va="bottom")

legend_handles = [
    mpatches.Patch(color='red', label='Highly Active (> μ + σ)'),
    mpatches.Patch(color='gray', label='Low Activity (< μ - σ)'),
    mpatches.Patch(color='skyblue', label='Normal')
]
plt.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig(f"{OUTPUT_SRC}/activity_index.png")

# CLeanup
shutil.rmtree(FACE_SAMPLE)