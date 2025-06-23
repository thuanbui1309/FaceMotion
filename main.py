import cv2
from ultralytics import YOLO

# Load mô hình YOLOv8 đã huấn luyện cho phát hiện khuôn mặt
model = YOLO("yolov8m-face.pt")  # hoặc yolov8m-face.pt tùy cấu hình máy

cap = cv2.VideoCapture(0)  # Mở webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf > 0.5:  # Chỉ hiển thị nếu độ tin cậy > 50%
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
