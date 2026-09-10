import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8 model (auto-downloads 'yolov8n.pt' on first run)
print("Loading YOLO model...")
model = YOLO("yolov8n.pt") 

# 2. Replace with your ESP32-CAM's IP address
ESP32_IP = "10.130.108.143"
stream_url = f"http://{ESP32_IP}/live"

print(f"Connecting to restricted ESP32-CAM stream at {stream_url}...")
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open the video stream.")
    print("Ensure the ESP32 is powered on and that this laptop's IP is 10.130.108.219.")
    exit()

print("Stream connected successfully! Press 'q' in the video window to quit.")

while True:
    # 3. Read a frame from the live stream
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from stream or connection was blocked/lost. Exiting...")
        break

    # 4. Run YOLO object detection on the frame
    results = model(frame, verbose=False)

    # 5. Draw bounding boxes and labels onto the image
    annotated_frame = results[0].plot()

    # 6. Display the live result window
    cv2.imshow("Secure ESP32-CAM YOLO Detection", annotated_frame)

    # 7. Press the 'q' key to safely quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()