import cv2
import face_recognition
import torch
import time
import numpy as np
from facenet_pytorch import InceptionResnetV1

# โหลด FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# โหลด YOLOv5 (ONNX) ด้วย OpenCV DNN
yolo_net = cv2.dnn.readNet(r"C:\CPE\PythonOpencvtest\yolov5-face.onnx")

yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# โหลดฐานข้อมูลใบหน้า
known_face_encodings = []
known_face_names = []

def add_face_to_database(image_path, name):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# เพิ่มข้อมูลใบหน้า
add_face_to_database("face_db/person1/img1.jpg", "beem")

# เปิดวิดีโอ
video_path = "video/5M.mp4"
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("❌ Error: Could not open video file!")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
total_time = video_length / fps

time_interval = 60  # 1 นาที
next_time_check = time_interval

print(f"✅ Total video time: {total_time / 60:.2f} minutes")

cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # ใช้ YOLOv5 ตรวจจับใบหน้า
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward()

    # ตรวจจับใบหน้าในเฟรม
    for detection in detections[0]:
        confidence = float(detection[4])
        if confidence > 0.5:
            x, y, w, h = map(int, detection[:4])
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            # แปลงใบหน้าเป็น RGB
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_face)
            
            if face_encoding:
                face_encoding = face_encoding[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                
                # แสดงผล
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
