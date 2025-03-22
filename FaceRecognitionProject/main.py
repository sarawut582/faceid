import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from facenet_pytorch import MTCNN
import mediapipe as mp

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_model(onnx_path):
    """โหลด ONNX Model และสร้าง TensorRT Engine"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # ตรวจสอบ Dynamic Shape
    input_tensor = network.get_input(0)
    if input_tensor.shape[0] == -1:
        print("⚠️ Detected Dynamic Batch Size, setting optimization profile...")
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, (1, 3, 160, 160), (4, 3, 160, 160), (8, 3, 160, 160))
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ Failed to serialize TensorRT engine!")
        return None

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)

onnx_path = "C:/CPE/PythonOpencvtest/facenet.onnx"
trt_engine = load_trt_model("C:/CPE/PythonOpencvtest/facenet_fixed.onnx")

if trt_engine:
    context = trt_engine.create_execution_context()
    print("✅ TensorRT Engine Loaded Successfully!")
else:
    print("❌ ไม่สามารถโหลด TensorRT engine ได้!")

mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

face_embeddings = np.load("face_database.npy", allow_pickle=True).item()

def recognize_face_with_trt(face_resized):
    """ ใช้ TensorRT ในการทำ inference สำหรับ FaceNet """
    # ตรวจสอบว่า face_resized มี 3 ช่อง (RGB/BGR)
    if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
        input_data = np.transpose(face_resized, (2, 0, 1)).astype(np.float32).flatten()
    else:
        print("❌ พบปัญหากับรูปภาพที่ส่งเข้าไป face_resized ต้องมี 3 ช่อง")
        return None

    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)

    output_data = np.empty([512], dtype=np.float32)  
    d_output = cuda.mem_alloc(output_data.nbytes)

    context.execute_v2([int(d_input), int(d_output)])

    cuda.memcpy_dtoh(output_data, d_output)
    return output_data

video_path = r"C:\CPE\PythonOpencvtest\video\5M.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("\nError: ไม่สามารถเปิดไฟล์วิดีโอได้!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face = frame[y:y + h, x:x + w]
            if face.size > 0:
                face_resized = cv2.resize(face, (160, 160))

                # ตรวจสอบการสร้าง face embedding
                face_embedding = recognize_face_with_trt(face_resized)

                if face_embedding is None:
                    continue  # หากไม่สามารถสร้าง face embedding ได้ ให้ข้ามไป

                name = "Unknown"
                min_distance = float("inf")

                for db_name, db_embedding in face_embeddings.items():
                    dist = np.linalg.norm(face_embedding - db_embedding)
                    print(f"Comparing with {db_name}, Distance: {dist:.2f}")  # Debugging output
                    if dist < min_distance:
                        min_distance = dist
                        name = db_name

                confidence = 1 - min_distance
                print(f"\n[✔] ตรวจพบใบหน้า: {name}, ความมั่นใจ: {confidence:.2f}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition + Liveness Detection with TensorRT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
