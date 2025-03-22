import face_recognition as face
import numpy as np
import cv2

# ใช้ r"..." หรือ / แทน \
video_capture = cv2.VideoCapture(r"C:\CPE\PythonOpencvtest\video\video_L_1.mp4")

L_image = face.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\LLL.jpg")
L_face_encoding = face.face_encodings(L_image)[0]

face_locations = []
face_encodings = []
face_names = []
face_percent = []
process_this_frame = True

known_face_encodings = [L_face_encoding]
known_face_names = ["Kittamat"]

while True:
    ret, frame = video_capture.read()
    if ret:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_names = []
        face_percent = []

        if process_this_frame:
            # แก้จาก face.face_location เป็น face.face_locations
            face_locations = face.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:  # แก้จาก face_encoding เป็น face_encodings
                face_distances = face.face_distance(known_face_encodings, face_encoding)
                best = np.argmin(face_distances)
                face_percent_value = 1 - face_distances[best]

                if face_percent_value >= 0.5:
                    name = known_face_names[best]
                    percent = round(face_percent_value * 100, 2)
                else:
                    name = "UNKNOWN"
                    percent = 0  # ย้าย face_percent.append(0) มาที่นี่

                face_names.append(name)
                face_percent.append(percent)  # เพิ่ม percent ลง list ที่ถูกต้อง

        for (top, right, bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            if name == "UNKNOWN":
                color = [46, 2, 209]
            else:
                color = [255, 102, 255]  # แก้สีให้ถูก (255,102,511 -> 255,102,255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left - 1, top - 30), (right + 1, top), color, cv2.FILLED)
            cv2.rectangle(frame, (left - 1, bottom), (right + 1, bottom + 30), color, cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "MATCH: " + str(percent) + "%", (left + 6, bottom), font, 0.6, (255, 255, 255), 1)

        process_this_frame = not process_this_frame

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video_capture.release()
cv2.destroyAllWindows()