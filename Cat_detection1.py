from flask import Flask, render_template, Response
import cv2
from threading import Thread, Lock
import time
import numpy as np
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# URL của luồng video (Thay bằng IP Camera của bạn)
stream_url = 'http://172.20.10.2:4747/video'
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Cấu hình bộ đệm để xử lý video mượt hơn

# Kiểm tra kích thước khung hình
ret, frame = cap.read()
if ret:
    height, width = frame.shape[:2]
else:
    width, height = 640, 480  # Giá trị mặc định nếu không đọc được khung hình

video_frame = None  # Lưu trữ khung hình hiện tại
lock = Lock()  # Đảm bảo đồng bộ khi truy cập khung hình

running = True  # Biến điều khiển vòng lặp
object_detected = False
video_writer = None
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Load mô hình MobileNet SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Khởi tạo background subtractor để phát hiện chuyển động
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Hàm xử lý luồng video
def camera_stream():
    global cap, video_frame, running, object_detected, video_writer

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Phát hiện chuyển động
        fgmask = fgbg.apply(frame)
        motion_detected = np.sum(fgmask) > 10000  # Ngưỡng phát hiện chuyển động

        # Nếu có chuyển động, thực hiện phát hiện đối tượng
        current_object_detected = False
        if motion_detected:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Ngưỡng tin cậy cao hơn
                    idx = int(detections[0, 0, i, 1])
                    if 0 <= idx < len(CLASSES):  # Kiểm tra tránh lỗi index
                        label = CLASSES[idx]
                        if label == "person":  # Chỉ nhận diện người
                            current_object_detected = True
                            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                            (startX, startY, endX, endY) = box.astype("int")

                            # Vẽ bounding box
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, f"{label}: {confidence * 100:.2f}%", (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Bắt đầu quay video nếu phát hiện người
        if current_object_detected and not object_detected:
            object_detected = True
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(output_folder, f"video_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
            print(f"Bắt đầu quay video: {video_filename}")

            # Chụp ảnh khi phát hiện người
            image_filename = os.path.join(output_folder, f"image_{timestamp}.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"Đã chụp ảnh: {image_filename}")

        # Dừng quay video nếu không còn người
        if not current_object_detected and object_detected:
            object_detected = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print("Dừng quay video")

        # Nếu đang quay video, ghi khung hình
        if object_detected and video_writer is not None:
            video_writer.write(frame)

        # Cập nhật khung hình
        with lock:
            video_frame = frame.copy()

# Hàm gửi khung hình đến client
def gen_frames():
    global video_frame, running
    while running:
        with lock:
            if video_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', video_frame)
            if not ret:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Định tuyến Flask để truyền luồng video
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Trang chính hiển thị video stream
@app.route('/')
def index():
    return render_template('index.html')

# Chạy luồng video trên thread riêng
thread = Thread(target=camera_stream, daemon=True)
thread.start()

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
