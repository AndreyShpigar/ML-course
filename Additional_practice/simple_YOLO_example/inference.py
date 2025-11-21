"""

"""
from ultralytics import YOLO
import cv2

# загрузка модели из репозитория https://github.com/YapaLab/yolo-face
model = YOLO("yolov11m_face.pt")
# model = YOLO("best.pt")
 
# инициализация камеры
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: could not open cam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = "YOLOv8 Face Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: cannot grab frame")
            break

        results = model(frame, conf=0.6, verbose=False)
        processed_frame = results[0].plot()
        cv2.imshow(window_name, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()