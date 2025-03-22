import os

file_path = "face_database.npy"

if os.path.exists(file_path):
    print(f"✅ พบไฟล์ที่: {os.path.abspath(file_path)}")
else:
    print("❌ ไม่พบไฟล์ face_database.npy! โปรดตรวจสอบพาธ")
