import cv2
import socket
import numpy as np
from ultralytics import YOLO


class MjpegStream:
    def __init__(self, url):
        host, path = url.removeprefix("http://").split("/", 1)
        self.socket = socket.create_connection((host, 80), timeout=5)
        self.socket.sendall(
            f"GET /{path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
        )
        self.buffer = b""
        headers = self._read_until(b"\r\n\r\n")
        if not headers.startswith(b"HTTP/1.0 200"):
            raise RuntimeError(f"Camera returned an unexpected response: {headers[:80]!r}")

    def _read_until(self, marker):
        while marker not in self.buffer:
            data = self.socket.recv(8192)
            if not data:
                raise RuntimeError("Camera stream closed before a complete response")
            self.buffer += data
        end = self.buffer.index(marker) + len(marker)
        result, self.buffer = self.buffer[:end], self.buffer[end:]
        return result

    def read(self):
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    jpeg = self.buffer[start:end + 2]
                    self.buffer = self.buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    return frame is not None, frame
            data = self.socket.recv(8192)
            if not data:
                return False, None
            self.buffer += data

    def release(self):
        self.socket.close()

# 1. Load the YOLOv8 model (this will auto-download 'yolov8n.pt' the first time)
print("Loading YOLO model...")
model = YOLO("yolov8n.pt") 

# 2. Your ESP32-CAM stream URL
# Notice the /live at the end - this pulls the raw MJPEG video
stream_url = "http://10.130.108.143/live"

print(f"Connecting to ESP32-CAM at {stream_url}...")
try:
    cap = MjpegStream(stream_url)
except (OSError, RuntimeError) as error:
    print("Error: Could not open the video stream.")
    print(error)
    print("Make sure your laptop is connected to 'vivo T4' and the ESP32 is powered on.")
    exit()

print("Stream connected successfully! Press 'q' in the video window to quit.")

while True:
    # 3. Read a frame from the live stream
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from stream or connection lost. Exiting...")
        break

    # 4. Run YOLO object detection on the frametr
    # verbose=False stops it from flooding your terminal with text every millisecond
    results = model(frame, verbose=False)

    # 5. Draw the bounding boxes and labels onto the image
    annotated_frame = results[0].plot()

    # 6. Display the live result on your screen
    cv2.imshow("ESP32-CAM YOLO Real-Time Detection", annotated_frame)

    # 7. Press the 'q' key to safely quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the window when done
cap.release()
cv2.destroyAllWindows()