import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import os

# โหลดโมเดล FaceNet
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# โฟลเดอร์ที่เก็บรูปภาพใบหน้า
dataset_path = r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset"  # เปลี่ยนเป็นโฟลเดอร์ของคุณ
if not os.path.exists(dataset_path):
    print(f"❌ ไม่พบโฟลเดอร์ '{dataset_path}' โปรดตรวจสอบอีกครั้ง")
    exit()

face_embeddings = {}

# อ่านโฟลเดอร์บุคคลจากฐานข้อมูล
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_folder):  # ตรวจสอบว่าเป็นโฟลเดอร์
        embeddings_list = []
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_folder, filename)

                # โหลดภาพและแปลงเป็น RGB
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (160, 160))
                img = np.transpose(img, (2, 0, 1)) / 255.0  # Normalize
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

                # ดึง Face Embedding
                with torch.no_grad():
                    embedding = facenet(img_tensor).numpy().flatten()
                    embeddings_list.append(embedding)

                print(f"✅ สร้าง embedding สำหรับ: {person_name}, รูป: {filename}")

        # ใช้ค่าเฉลี่ยของ embeddings ในการสร้างฐานข้อมูล
        face_embeddings[person_name] = np.mean(embeddings_list, axis=0)

# บันทึกฐานข้อมูลลงไฟล์ .npy
np.save("face_database.npy", face_embeddings)
print("📂 บันทึกฐานข้อมูลใบหน้าเสร็จสิ้น! พบทั้งหมด", len(face_embeddings), "คน")
