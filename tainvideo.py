import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import time
from scipy.spatial.distance import cosine

# โหลด YOLOv5 โมเดล (หรือเปลี่ยนเป็น YOLOv8 ได้)
yolo_model_path = "yolov5s.pt"  # หรือเปลี่ยนเป็น .engine หากใช้ TensorRT
yolo_model = YOLO(yolo_model_path)

# โหลด FaceNet สำหรับดึงคุณลักษณะใบหน้า
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# 📝 ฐานข้อมูลใบหน้า (ต้องเพิ่มข้อมูลบุคคลที่รู้จักล่วงหน้า)
known_faces = {
    "Alice": np.random.rand(512),  # ควรใช้ embeddings จริงจาก FaceNet
    "Bob": np.random.rand(512),
    "Charlie": np.random.rand(512)
}
threshold = 0.5  # ค่าตัดสินว่า embeddings ใกล้กันแค่ไหน (ค่าใกล้ 0 คือเหมือนกัน)

# เปิดวิดีโอ
video_path = r"C:\CPE\PythonOpencvtest\video\5M.mp4"  # เปลี่ยนเป็นพาธของไฟล์วิดีโอ
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video file!")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # วิดีโอจบ

    # YOLO ตรวจจับวัตถุ
    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดกล่อง
            conf = box.conf[0].item()  # ความมั่นใจ
            cls = int(box.cls[0].item())  # หมวดหมู่

            # ถ้าเป็น "person" (YOLO class ID 0)
            if cls == 0 and conf > 0.6:
                face = frame[y1:y2, x1:x2]

                if face.size > 0:
                    # แปลงภาพเป็น RGB และย่อขนาด
                    face = cv2.resize(face, (160, 160))
                    face = np.transpose(face, (2, 0, 1)) / 255.0  # Normalize
                    face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0)

                    # ใช้ FaceNet ดึงคุณลักษณะใบหน้า
                    with torch.no_grad():
                        face_embedding = facenet(face_tensor).numpy().flatten()

                    # 🔍 เปรียบเทียบกับฐานข้อมูลใบหน้า
                    best_match = "Unknown"
                    best_score = float("inf")

                    for name, known_embedding in known_faces.items():
                        score = cosine(face_embedding, known_embedding)
                        if score < best_score and score < threshold:
                            best_match = name
                            best_score = score

                    # 🔹 บันทึกข้อมูลลง console
                    print(f"📌 Detected: {best_match} (Score: {best_score:.4f})")

                    # 🔹 วาดกรอบและแสดงชื่อบุคคล
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงผลวิดีโอ
    cv2.imshow("YOLO + FaceNet", frame)
    
    # ปิดโปรแกรมเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# คำนวณ FPS
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"🎥 FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
