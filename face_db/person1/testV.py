import cv2

# เปิดวิดีโอ
video_path = r"C:\CPE\PythonOpencvtest\video\video_beem_1.mp4"
video_capture = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดวิดีโอสำเร็จหรือไม่
if not video_capture.isOpened():
    print("❌ Error: Could not open video file. Check the file path!")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break  # หยุดเมื่อวิดีโอหมด

    # แสดงผลวิดีโอ
    cv2.imshow('Test Video', frame)

    # กด 'q' เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการประมวลผล
video_capture.release()
cv2.destroyAllWindows()
