import cv2
import dlib
import face_recognition


#Load Known face encodings and names
known_face_encodings = []
known_face_names = []

# Load Known faces and their names here
known_person1_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\person1\beem01.jpg")

known_person2_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\person2\cho01.jpg")
known_person3_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\person3\L01.jpg")

known_person1_encodeing = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encodeing = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encodeing = face_recognition.face_encodings(known_person3_image)[0]

known_face_encodings.append(known_person1_encodeing)
known_face_encodings.append(known_person2_encodeing)
known_face_encodings.append(known_person3_encodeing)

known_face_names.append("Beem")
known_face_names.append("Chokun")
known_face_names.append("L")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        #check if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "ghost"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw box around the face and label with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()





