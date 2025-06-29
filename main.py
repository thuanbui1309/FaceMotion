import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import shutil

# Params
MODEL = "models/yolov8n-face.pt"
VIDEO = "data/demo.mp4"
OUTPUT_SRC = "runs"
FACE_SAMPLE = "face_samples"
KNOWN_FACES = "known_faces"
DIST_THRESHOLD = 50
MAX_HISTORY = 30
EMA_ALPHA = 0.2
WARMUP_FRAMES = 30

# Init
model = YOLO(MODEL)
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=True, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

prev_faces = {}
next_id = 0
track_history = {}
motion_log = {}
ema_state = {}
ema_values_accum = []
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

        x1, y1, x2, y2 = box
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0 and frame_idx % 50 == 0:
            save_dir = f"{FACE_SAMPLE}/id_{matched_id}"
            os.makedirs(save_dir, exist_ok=True)
            count = len(os.listdir(save_dir))
            cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_crop)

        if matched_id in prev_faces:
            px, py = prev_faces[matched_id]
            dx, dy = cx - px, cy - py
            movement = np.sqrt(dx ** 2 + dy ** 2)
        else:
            movement = 0.0

        prev_ema = ema_state.get(matched_id, 0.0)
        curr_ema = EMA_ALPHA * movement + (1 - EMA_ALPHA) * prev_ema
        ema_state[matched_id] = curr_ema

        if matched_id not in motion_log:
            motion_log[matched_id] = []
        motion_log[matched_id].append((frame_idx, movement))

        if frame_idx < WARMUP_FRAMES:
            ema_values_accum.append(curr_ema)
            mu_ema = 0
            sigma_ema = 0
        else:
            mu_ema = np.mean(ema_values_accum)
            sigma_ema = np.std(ema_values_accum)

        if frame_idx < WARMUP_FRAMES:
            level = "Warming"
            box_color = (128, 128, 128)
        elif curr_ema > mu_ema + sigma_ema:
            level = "Active"
            box_color = (0, 0, 255)
        elif curr_ema < mu_ema - sigma_ema:
            level = "Inactive"
            box_color = (150, 150, 150)
        else:
            level = "Normal"
            box_color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"ID {matched_id} ({level})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

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

# ========== Encode known_faces ==========
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

# ========== Match sampled faces to known ==========
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

# ========== Store movement & EMA ==========
records = []
for id_, log in motion_log.items():
    prev_ema = 0.0
    for frame_idx, movement in log:
        ema = EMA_ALPHA * movement + (1 - EMA_ALPHA) * prev_ema
        records.append((frame_idx, id_, movement, ema))
        prev_ema = ema

df = pd.DataFrame(records, columns=["frame", "id", "movement", "ema"])
df["name"] = df["id"].map(id_to_name)
df.to_csv(f"{OUTPUT_SRC}/motion_activity.csv", index=False)

# ========== Plot Rolling Mean ==========
plt.figure(figsize=(12, 6))
for name in df["name"].unique():
    df_pid = df[df["name"] == name].sort_values("frame")
    plt.plot(df_pid["frame"], df_pid["movement"].rolling(5, min_periods=1).mean(), label=name)
plt.axhline(df["movement"].mean(), color='red', linestyle='--', label="Avg Movement")
plt.title("Movement Activity Over Time (Rolling Avg)")
plt.xlabel("Frame")
plt.ylabel("Movement")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_SRC}/motion_plot_rolling.png")

# ========== Plot EMA ==========
plt.figure(figsize=(12, 6))
for name in df["name"].unique():
    df_pid = df[df["name"] == name].sort_values("frame")
    plt.plot(df_pid["frame"], df_pid["ema"], label=name)
plt.axhline(df["ema"].mean(), color='red', linestyle='--', label="Avg EMA")
plt.title("Movement Activity Over Time (EMA Smoothed)")
plt.xlabel("Frame")
plt.ylabel("EMA")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_SRC}/motion_plot_ema.png")

# ========== Interactive Plot ==========
fig = go.Figure()
for name in df["name"].unique():
    df_pid = df[df["name"] == name].sort_values("frame")
    fig.add_trace(go.Scatter(x=df_pid["frame"], y=df_pid["movement"].rolling(5, min_periods=1).mean(),
                             mode='lines', name=f"{name} (Rolling)", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_pid["frame"], y=df_pid["ema"],
                             mode='lines', name=f"{name} (EMA)", line=dict(dash='solid')))
fig.update_layout(title="Interactive Comparison: EMA vs Rolling Avg",
                  xaxis_title="Frame", yaxis_title="Movement", template="plotly_white")
fig.write_html(f"{OUTPUT_SRC}/interactive_motion_comparison.html")

# ========== Activity Index ==========
activity_df = df.groupby("name")["ema"].mean().reset_index()
activity_df.rename(columns={"ema": "activity_index"}, inplace=True)
activity_df = activity_df.sort_values("activity_index", ascending=False)

mu = activity_df["activity_index"].mean()
sigma = activity_df["activity_index"].std()
colors = []
for val in activity_df["activity_index"]:
    if val > mu + sigma:
        colors.append("red")
    elif val < mu - sigma:
        colors.append("gray")
    else:
        colors.append("skyblue")

plt.figure(figsize=(10, 5))
bars = plt.bar(activity_df["name"], activity_df["activity_index"], color=colors)
plt.title("Activity Index per Student (EMA)")
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

# ========== Cleanup ==========
shutil.rmtree(FACE_SAMPLE)
