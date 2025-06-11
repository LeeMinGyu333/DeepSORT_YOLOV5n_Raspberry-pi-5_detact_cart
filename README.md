# DeepSORT_YOLOV5n_Raspberry-pi-5_detact_cart

communication: UDP(Raspberry pi 5 -> PC) to use GStreamer

#installation.(Ubuntu)
sudo apt-get install v4l-utils


#1. Raspberry Pi code (transfer webcam screen)
# Transfer H264(compacted) to use Gstreamer
sudo apt update
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good


# in Raspberry Pi Terminal,  <PC_IP>= YOUR PC IP
gst-launch-1.0 v4l2src device=/dev/video0 ! \
  video/x-raw,width=640,height=480,framerate=30/1 ! \
  videoconvert ! \
  x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! \
  rtph264pay config-interval=1 pt=96 ! \
  udpsink host=<PC_IP> port=5000


#2. Windows PC code (YOLO5n + DeepSORT)
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import random
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

#YOLO model loaded
model=torch.hub.load('ultralytics/yolov5n','custom',path='C:\Users\KOSTA\Downloads\best2.pt',device='cpu',force_reload=True)
print("YOLOv5 model load complete")

#DeepSort init
trackeer=DeepSort(max_age=20)

colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(50)]


#OpenCV pipeline UDP stream to Raspberry pi
gst_pipeline=(
    "udpsrc port=5000 caps=\"application/x-rtp, media=video, "
    "encoding-name=H264, payload=96\" ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
) 

cap=cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("video stream opening failed.")
    exit()

detection_threshold=0.5

target_id=None   # first detected person ID
target_distance=100 # distance with person & cart(px)

while True:
    ret, frame=cap.read()
    if not ret:
        print("FRAME READING FAIELD")
        break

    results=model(frame)

    detections=[]
    for result in results.xyxy[0].tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if score > detection_threshold:
            detections.append(([x1, y1, x2-x1, y2-y1],score, 'person'))

    tracks=tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        color = colors[int(track_id) % len(colors)]
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 3)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        #first detected person setting
        if target_id is None:
            target_id = track_id
            print(f" target ID setting: {target_id}")

        #indicate TARGET
        if track_id == target_id:
            target_center_x = (l+r)/2
            target_center_y = (t+b)/2
            # Control this cart to use these two centers.

            cv2.circle(frame,(int(target_center_x), int(target_center_y)), 5, (0,255,0), -1)

    cv2.imshow('Real-time Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
