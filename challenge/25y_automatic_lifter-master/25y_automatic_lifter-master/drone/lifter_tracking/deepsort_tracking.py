import os
import cv2
import threading
import queue
import time
from ultralytics import YOLO
from djitellopy import Tello
from deep_sort_realtime.deepsort_tracker import DeepSort

# =============== 설정 ================
TARGET_CLASS   = 'lifter'
CONF_THRESHOLD = 0.4

FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
CENTER_X       = FRAME_WIDTH  // 2
CENTER_Y       = FRAME_HEIGHT // 2
CENTER_MARGIN  = 90  # Deadzone

MIN_BOX_HEIGHT = 120
MAX_BOX_HEIGHT = 180

Kp_yaw = 0.15
MAX_YAW = 40
MAX_FB  = 30

SMOOTH_ALPHA = 0.85

WEIGHTS_PATH   = "../krri_models/lifter_person.pt"
QUEUE_MAXSIZE  = 10
# ====================================

# YOLO 모델 로드
model = YOLO(WEIGHTS_PATH)
names = model.names
lifter_class_id = [k for k, v in names.items() if v == 'lifter'][0]

# Deep SORT
tracker = DeepSort(max_age=15)

# DJI Tello
tello = Tello()
tello.connect()
print("배터리:", tello.get_battery(), "%")
tello.streamon()

# 프레임 수집
frame_queue = queue.LifoQueue(maxsize=QUEUE_MAXSIZE)
stop_event = threading.Event()

def frame_reader_loop():
    fr = tello.get_frame_read()
    while not stop_event.is_set():
        frame = fr.frame
        if frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.005)

reader_thread = threading.Thread(target=frame_reader_loop, daemon=True)
reader_thread.start()

# 초기값
prev_yaw = 0
prev_fb  = 0

frame_idx = 0

# 이륙
tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)

try:
    while True:

        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            tello.send_rc_control(0, 0, 0, 0)
            continue

        # ---------- YOLO 탐지 ----------
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        det = results[0]

        lifter_detections = []
        target_box = None

        for box in det.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if cls == lifter_class_id:
                lifter_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "lifter"))
                target_box = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Lifter {conf:.2f}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ---------- DeepSORT ----------
        tracks = tracker.update_tracks(lifter_detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.putText(frame, f'ID:{track.track_id}', (x1, y1-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ---------- RC 제어 ----------
        yaw = 0
        fb  = 0

        if target_box:
            x1, y1, x2, y2 = target_box
            cx = (x1 + x2) // 2
            box_height = y2 - y1

            err_x = cx - CENTER_X
            if abs(err_x) < CENTER_MARGIN:
                err_x = 0
            yaw = int(max(min(Kp_yaw * err_x, MAX_YAW), -MAX_YAW))

            if box_height < MIN_BOX_HEIGHT:
                fb = 10
            elif box_height > MAX_BOX_HEIGHT:
                fb = -10
            else:
                fb = 0
        else:
            yaw, fb = 0, 0

        # smoothing
        fb  = int(SMOOTH_ALPHA * prev_fb  + (1-SMOOTH_ALPHA) * fb)
        yaw = int(SMOOTH_ALPHA * prev_yaw + (1-SMOOTH_ALPHA) * yaw)

        tello.send_rc_control(0, fb, 0, yaw)
        prev_fb, prev_yaw = fb, yaw

        # ---------- 화면 표시 ----------
        cv2.line(frame, (CENTER_X - CENTER_MARGIN, CENTER_Y),
                        (CENTER_X + CENTER_MARGIN, CENTER_Y), (255,255,255), 1)

        cv2.putText(frame, f"RC fb:{fb} yaw:{yaw}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Drone View (YOLO + DeepSORT ID)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        frame_idx += 1

except KeyboardInterrupt:
    print("Ctrl+C detected. Safe shutdown...")

finally:
    stop_event.set()
    try:
        reader_thread.join(timeout=1.0)
    except:
        pass
    try:
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)
        tello.land()
    except Exception as e:
        print("착륙 중 예외:", e)
    tello.streamoff()
    cv2.destroyAllWindows()
