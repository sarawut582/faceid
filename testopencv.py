import cv2
import dlib
from deepface import DeepFace

# ใช้กล้องเว็บแคมแทน RTSP
cap = cv2.VideoCapture(0)  # 0 หมายถึงกล้องหลักของโน้ตบุ๊ก

detector = dlib.get_frontal_face_detector()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_crop = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.find(img_path=face_crop, db_path="face_db", model_name="VGG-Face")
            if len(result) > 0:
                cv2.putText(frame, "Match Found!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except:
            cv2.putText(frame, "No Match", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
