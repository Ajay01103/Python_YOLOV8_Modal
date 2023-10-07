from ultralytics import YOLO
import cv2

# load video
model = YOLO('yolov8n.pt')

video_path = './test.mp4'

cap = cv2.VideoCapture(video_path)

ret = True

# Read Frames

while ret:
    ret, frame = cap.read()

    # Detect Objects

    results = model.track(frame, persist=True)

    # Plot Results
    frame_ = results[0].plot()

    # Visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break