import os
import sys
import time
import queue
import threading
import logging

import cv2
import dxcam
import numpy as np
import torch
import win32api
import win32con
import win32gui
import keyboard
from ultralytics import YOLO

class Config:
    MONITOR_WIDTH = 320
    MONITOR_HEIGHT = 320
    MODEL_PATH = "overwatch.pt"
    ENGINE_PATH = "overwatch_tensorrt.engine"
    CONF_THRESHOLD = 0.45
    TARGET_CLASS = 0
    MAX_DISTANCE = 450
    DEBUG_WINDOW_NAME = "Overwatch AI SoftAim v5 | F2: Toggle AI"
    SENSITIVITY = 0.8
    SOFT_AIM_FACTOR = 1.2
    IMAGE_QUEUE_SIZE = 1
    RESULT_QUEUE_SIZE = 1
    ALWAYS_ON_TOP = True
    PREDICT_STRENGTH = 3.0
    SMOOTHING_ALPHA = 0.1

stop_event = threading.Event()
AI_ACTIVE = False
SHOW_DEBUG_WINDOW = True
IMAGE_QUEUE = queue.Queue(maxsize=Config.IMAGE_QUEUE_SIZE)
RESULT_QUEUE = queue.Queue(maxsize=Config.RESULT_QUEUE_SIZE)
PREV_TARGET_POS = None
PREV_TARGET_TIME = None
SMOOTHED_POS = None

def get_primary_monitor_resolution():
    return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

def set_window_always_on_top(window_title):
    try:
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1:
            hwnd = win32gui.FindWindow(None, window_title)
            if hwnd:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST,
                                      0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    except Exception as e:
        logging.warning("Cannot set always on top: %s", e)

def move_mouse(x_offset, y_offset):
    x_move = int(x_offset / Config.SOFT_AIM_FACTOR * Config.SENSITIVITY)
    y_move = int(y_offset / Config.SOFT_AIM_FACTOR * Config.SENSITIVITY)
    if x_move != 0 or y_move != 0:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_move, y_move, 0, 0)

def convert_to_tensorrt():
    if os.path.exists(Config.ENGINE_PATH):
        logging.info("TensorRT engine exists. Skipping conversion.")
        return True
    if not os.path.exists(Config.MODEL_PATH):
        logging.warning("Model not found. Using PyTorch.")
        return False
    try:
        model = YOLO(Config.MODEL_PATH)
        engine_path = model.export(
            format="engine",
            half=True,
            imgsz=[Config.MONITOR_HEIGHT, Config.MONITOR_WIDTH],
            device="0",
            simplify=True,
            workspace=8,
            iou=0.6,
            nms=True,
        )
        if engine_path and os.path.exists(engine_path):
            if engine_path != Config.ENGINE_PATH:
                if os.path.exists(Config.ENGINE_PATH):
                    os.remove(Config.ENGINE_PATH)
                os.rename(engine_path, Config.ENGINE_PATH)
            logging.info(f"TensorRT engine created: {Config.ENGINE_PATH}")
            return True
    except Exception as e:
        logging.exception("TensorRT conversion failed.")
    return False

class CaptureThread(threading.Thread):
    def __init__(self, left, top):
        super().__init__(daemon=True)
        self.left = left
        self.top = top

    def run(self):
        try:
            cam = dxcam.create(
                output_idx=0,
                region=(self.left, self.top,
                        self.left + Config.MONITOR_WIDTH,
                        self.top + Config.MONITOR_HEIGHT),
                output_color="BGR",
                max_buffer_len=1,
            )
            cam.start()
        except Exception as e:
            logging.error("DXCam init failed: %s", e)
            return

        while not stop_event.is_set():
            frame = cam.get_latest_frame()
            if frame is not None:
                if IMAGE_QUEUE.full():
                    try: IMAGE_QUEUE.get_nowait()
                    except queue.Empty: pass
                IMAGE_QUEUE.put(frame)
            time.sleep(0.001)
        try:
            cam.stop()
        except: pass

class InferenceThread(threading.Thread):
    def __init__(self, model, device):
        super().__init__(daemon=True)
        self.model = model
        self.device = device

    def run(self):
        while not stop_event.is_set():
            try:
                img = IMAGE_QUEUE.get(timeout=0.5)
            except queue.Empty:
                continue

            if not AI_ACTIVE:
                if RESULT_QUEUE.full():
                    try: RESULT_QUEUE.get_nowait()
                    except queue.Empty: pass
                RESULT_QUEUE.put((None, img))
                continue

            try:
                with torch.no_grad():
                    results_list = self.model.predict(
                        source=img,
                        stream=False,
                        verbose=False,
                        conf=Config.CONF_THRESHOLD,
                        device=self.device
                    )
                detections = results_list[0] if results_list else None
                if RESULT_QUEUE.full():
                    try: RESULT_QUEUE.get_nowait()
                    except queue.Empty: pass
                RESULT_QUEUE.put((detections, img))
            except Exception:
                time.sleep(0.01)

def predict_target(current_pos):
    global PREV_TARGET_POS, PREV_TARGET_TIME, SMOOTHED_POS
    if PREV_TARGET_POS is None:
        PREV_TARGET_POS = current_pos
        PREV_TARGET_TIME = time.time()
        SMOOTHED_POS = current_pos
        return current_pos

    now = time.time()
    dt = now - PREV_TARGET_TIME
    if dt <= 0:
        return SMOOTHED_POS

    dx = current_pos[0] - PREV_TARGET_POS[0]
    dy = current_pos[1] - PREV_TARGET_POS[1]

    pred_x = current_pos[0] + dx / dt * 0.016 * Config.PREDICT_STRENGTH
    pred_y = current_pos[1] + dy / dt * 0.016 * Config.PREDICT_STRENGTH

    smoothed_x = SMOOTHED_POS[0] * (1 - Config.SMOOTHING_ALPHA) + pred_x * Config.SMOOTHING_ALPHA
    smoothed_y = SMOOTHED_POS[1] * (1 - Config.SMOOTHING_ALPHA) + pred_y * Config.SMOOTHING_ALPHA

    PREV_TARGET_POS = current_pos
    PREV_TARGET_TIME = now
    SMOOTHED_POS = (int(smoothed_x), int(smoothed_y))

    return SMOOTHED_POS

def aim_at_target(detections, screen_center_x, screen_center_y, monitor_left, monitor_top):
    if not detections or detections.boxes.data.numel() == 0:
        return None

    data = detections.boxes.data.cpu().numpy()
    valid = data[(data[:,5]==Config.TARGET_CLASS) & (data[:,4]>=Config.CONF_THRESHOLD)]
    if valid.size == 0:
        return None

    x1,y1,x2,y2 = valid[:,0:4].T
    cx = ((x1+x2)/2).astype(int)
    cy = (y1 + (y2 - y1)/6).astype(int)

    dist = np.sqrt((cx - screen_center_x)**2 + (cy - screen_center_y)**2)
    close_idx = np.where(dist <= Config.MAX_DISTANCE)[0]
    if close_idx.size == 0:
        return None

    best = close_idx[np.argmin(dist[close_idx])]
    target_x = int(cx[best])
    target_y = int(cy[best])

    pred_x, pred_y = predict_target((target_x, target_y))

    global_x = monitor_left + pred_x
    global_y = monitor_top + pred_y

    cur_x, cur_y = win32api.GetCursorPos()
    delta_x = global_x - cur_x
    delta_y = global_y - cur_y
    move_mouse(delta_x, delta_y)

    return (pred_x, pred_y)

def toggle_ai(e=None):
    global AI_ACTIVE
    AI_ACTIVE = not AI_ACTIVE
    logging.info("AI Toggled: %s", "ACTIVE" if AI_ACTIVE else "INACTIVE")

def main():
    global SHOW_DEBUG_WINDOW

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    screen_w, screen_h = get_primary_monitor_resolution()
    monitor_left = (screen_w - Config.MONITOR_WIDTH)//2
    monitor_top = (screen_h - Config.MONITOR_HEIGHT)//2
    center_x = Config.MONITOR_WIDTH//2
    center_y = Config.MONITOR_HEIGHT//2

    tensorrt_ready = convert_to_tensorrt()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = Config.ENGINE_PATH if tensorrt_ready else Config.MODEL_PATH
    if not os.path.exists(model_path):
        logging.error("Model not found: %s", model_path)
        sys.exit(1)

    model = YOLO(model_path, task="detect")
    logging.info("Model loaded (%s)", "TensorRT" if tensorrt_ready else "PyTorch")
    logging.info("Device: %s", device)

    cap_thread = CaptureThread(monitor_left, monitor_top)
    infer_thread = InferenceThread(model, device)
    cap_thread.start()
    infer_thread.start()

    keyboard.on_press_key("f2", lambda e: toggle_ai(e))
    logging.info("F2: Toggle AI")

    window_set_topmost = False

    try:
        while not stop_event.is_set():
            try:
                detections, img = RESULT_QUEUE.get(timeout=0.5)
            except queue.Empty:
                continue

            if SHOW_DEBUG_WINDOW and img is not None:
                if not window_set_topmost:
                    set_window_always_on_top(Config.DEBUG_WINDOW_NAME)
                    window_set_topmost = True

                target_vis = None
                if AI_ACTIVE and detections:
                    target_vis = aim_at_target(detections, center_x, center_y, monitor_left, monitor_top)

                if detections and AI_ACTIVE:
                    for det in detections.boxes.data.cpu().numpy():
                        x1,y1,x2,y2 = det[0:4].astype(int)
                        conf = det[4]; cls=int(det[5])
                        if cls==Config.TARGET_CLASS and conf>=Config.CONF_THRESHOLD:
                            color=(255,0,0)
                            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                            cv2.putText(img,f"{conf:.2f}",(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

                if target_vis:
                    cv2.circle(img,target_vis,7,(0,0,255),-1)
                    cv2.line(img,(center_x,center_y),target_vis,(0,255,0),2)

                status_txt = "ACTIVE" if AI_ACTIVE else "INACTIVE"
                color_status = (0,255,0) if AI_ACTIVE else (0,0,255)
                cv2.putText(img,f"AI: {status_txt}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color_status,2)
                cv2.line(img,(center_x-10,center_y),(center_x+10,center_y),(255,255,255),2)
                cv2.line(img,(center_x,center_y-10),(center_x,center_y+10),(255,255,255),2)

                cv2.imshow(Config.DEBUG_WINDOW_NAME,img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            elif AI_ACTIVE and detections:
                aim_at_target(detections, center_x, center_y, monitor_left, monitor_top)

            time.sleep(0.002)

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        cap_thread.join(timeout=2)
        infer_thread.join(timeout=2)
        if SHOW_DEBUG_WINDOW:
            try: cv2.destroyAllWindows()
            except: pass
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    main()
