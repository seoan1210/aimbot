import cv2
import numpy as np
import time
import win32api
import win32con
import keyboard
from mss import mss
import torch
from ultralytics import YOLO
import os
import win32gui
import sys
import threading
import queue 

MONITOR_WIDTH = 320
MONITOR_HEIGHT = 320
MODEL_PATH = 'overwatch.pt'
ENGINE_PATH = 'overwatch_tensorrt.engine'
CONF_THRESHOLD = 0.5
TARGET_CLASS = 0
SMOOTHING = 1.2
MAX_DISTANCE = 450
DEBUG_WINDOW_NAME = "Overwatch AI | F2: Toggle AI | F3: Toggle Mode"
PREDICTION_FACTOR = 0.04

ACCUMULATED_X_ERROR = 0.0
ACCUMULATED_Y_ERROR = 0.0
HUMAN_MODE_ACTIVE = False
AI_ACTIVE = False
PREV_TARGET_ON_SCREEN = None
SHOW_DEBUG_WINDOW = True
PREV_TARGET_GLOBAL_POS = None 
PREV_TARGET_TIME = None        

IMAGE_QUEUE = queue.Queue(maxsize=1) 
RESULT_QUEUE = queue.Queue(maxsize=1) 

def get_primary_monitor_resolution():
    return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

def move_mouse(x_offset, y_offset, human_mode=False):
    global ACCUMULATED_X_ERROR, ACCUMULATED_Y_ERROR
    
    if human_mode:
        SMOOTH_FACTOR = 20
        x_target_move_float = x_offset / SMOOTH_FACTOR
        y_target_move_float = y_offset / SMOOTH_FACTOR
        
        threshold = 4
        if abs(x_offset) < threshold and abs(y_offset) < threshold:
            ACCUMULATED_X_ERROR *= 0.9
            ACCUMULATED_Y_ERROR *= 0.9
            return
            
    else:
        sensitivity = 0.15
        
        x_target_move_float = x_offset * sensitivity
        y_target_move_float = y_offset * sensitivity

        threshold = 7
        if abs(x_offset) <= threshold and abs(y_offset) <= threshold:
            ACCUMULATED_X_ERROR *= 0.9
            ACCUMULATED_Y_ERROR *= 0.9
            return

    x_move_with_error = x_target_move_float + ACCUMULATED_X_ERROR
    y_move_with_error = y_target_move_float + ACCUMULATED_Y_ERROR

    x_move = int(x_move_with_error)
    y_move = int(y_move_with_error)
    
    ACCUMULATED_X_ERROR = x_move_with_error - x_move
    ACCUMULATED_Y_ERROR = y_move_with_error - y_move

    if x_move != 0 or y_move != 0:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_move, y_move, 0, 0)

def convert_to_tensorrt():
    if os.path.exists(ENGINE_PATH):
        print("TensorRT Engine found. Skipping conversion.")
        return True
    
    print("Attempting to convert model to TensorRT Engine with FP16 optimization...")
    try:
        model = YOLO(MODEL_PATH)
        
        success_path = model.export(
            format='engine', 
            half=True,       
            imgsz=[MONITOR_HEIGHT, MONITOR_WIDTH], 
            device='0',
            simplify=True,
            workspace=8,     
            iou=0.6,         
            nms=True,        
        )
        
        if success_path and os.path.exists(success_path):
            if success_path != ENGINE_PATH: 
                if os.path.exists(ENGINE_PATH): 
                    os.remove(ENGINE_PATH)
                os.rename(success_path, ENGINE_PATH)
            print(f"TensorRT Engine successfully created at {ENGINE_PATH}")
            return True
        
        print("TensorRT conversion failed or path incorrect.")
        return False
            
    except Exception as e: 
        print(f"TensorRT Conversion Error: {e}")
        return False

def capture_screen(monitor_left, monitor_top):
    with mss() as sct:
        monitor_config = {
            "top": monitor_top, 
            "left": monitor_left, 
            "width": MONITOR_WIDTH, 
            "height": MONITOR_HEIGHT
        }
        return cv2.cvtColor(np.array(sct.grab(monitor_config)), cv2.COLOR_BGRA2BGR) 

def aim_at_target(detections_data, screen_center_x, screen_center_y, monitor_left, monitor_top):
    global PREV_TARGET_ON_SCREEN, PREV_TARGET_GLOBAL_POS, PREV_TARGET_TIME, PREDICTION_FACTOR

    current_time = time.time() 

    if not detections_data or detections_data.boxes.data.numel() == 0:
        PREV_TARGET_ON_SCREEN = None
        PREV_TARGET_GLOBAL_POS = None 
        PREV_TARGET_TIME = None
        return None
    
    detections = detections_data.boxes.data.cpu().numpy()
    valid_detections = detections[(detections[:, 5] == TARGET_CLASS) & (detections[:, 4] >= CONF_THRESHOLD)]
    
    if valid_detections.size == 0:
        PREV_TARGET_ON_SCREEN = None
        PREV_TARGET_GLOBAL_POS = None
        PREV_TARGET_TIME = None
        return None
    
    x1, y1, x2, y2 = valid_detections[:, 0:4].T
    target_center_x = ((x1 + x2) / 2).astype(int)
    target_center_y = (y1 + (y2 - y1) / 6).astype(int) 

    distances = np.sqrt((target_center_x - screen_center_x) ** 2 + (target_center_y - screen_center_y) ** 2)
    close_targets_indices = np.where(distances <= MAX_DISTANCE)[0]
    
    if close_targets_indices.size == 0:
        PREV_TARGET_ON_SCREEN = None
        PREV_TARGET_GLOBAL_POS = None
        PREV_TARGET_TIME = None
        return None
    
    best_target_index = close_targets_indices[np.argmin(distances[close_targets_indices])]
    
    target_x = target_center_x[best_target_index]
    target_y = target_center_y[best_target_index]
    
    current_global_x = monitor_left + target_x
    current_global_y = monitor_top + target_y

    predict_x_offset = 0
    predict_y_offset = 0

    if PREV_TARGET_GLOBAL_POS and PREV_TARGET_TIME:
        delta_time = current_time - PREV_TARGET_TIME
        
        delta_x = current_global_x - PREV_TARGET_GLOBAL_POS[0]
        delta_y = current_global_y - PREV_TARGET_GLOBAL_POS[1]

        PREDICTION_TIME = PREDICTION_FACTOR 
        
        if delta_time > 0.001: 
            predict_x_offset = int(delta_x / delta_time * PREDICTION_TIME)
            predict_y_offset = int(delta_y / delta_time * PREDICTION_TIME)

        if abs(predict_x_offset) > 50 or abs(predict_y_offset) > 50:
            predict_x_offset = 0
            predict_y_offset = 0

    final_target_x = target_x + predict_x_offset
    final_target_y = target_y + predict_y_offset
    
    if PREV_TARGET_ON_SCREEN and SMOOTHING > 1.0:
        prev_x, prev_y = PREV_TARGET_ON_SCREEN
        target_x_smooth = int(prev_x + (final_target_x - prev_x) / SMOOTHING)
        target_y_smooth = int(prev_y + (final_target_y - prev_y) / SMOOTHING)
    else:
        target_x_smooth, target_y_smooth = int(final_target_x), int(final_target_y)

    current_mouse_x, current_mouse_y = win32api.GetCursorPos()
    
    global_target_x = monitor_left + target_x_smooth
    global_target_y = monitor_top + target_y_smooth

    delta_x = global_target_x - current_mouse_x
    delta_y = global_target_y - current_mouse_y
    
    move_mouse(delta_x, delta_y, human_mode=HUMAN_MODE_ACTIVE)
    
    PREV_TARGET_ON_SCREEN = (target_x_smooth, target_y_smooth)
    PREV_TARGET_GLOBAL_POS = (current_global_x, current_global_y) 
    PREV_TARGET_TIME = current_time                               

    return (target_x_smooth, target_y_smooth)

def set_window_always_on_top(window_title):
    try:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    except Exception:
        pass 

def capture_thread_func(monitor_left, monitor_top):
    while True:
        img = capture_screen(monitor_left, monitor_top)
        
        if IMAGE_QUEUE.full():
            try:
                IMAGE_QUEUE.get_nowait()
            except queue.Empty:
                pass
        
        IMAGE_QUEUE.put(img)

def inference_thread_func(model, device_to_use):
    while True:
        try:
            img = IMAGE_QUEUE.get(timeout=1) 
            
            if not AI_ACTIVE:
                if RESULT_QUEUE.full(): 
                    RESULT_QUEUE.get_nowait()
                RESULT_QUEUE.put((None, img))
                time.sleep(0.001)
                continue

            with torch.no_grad():
                results_list = model.predict(
                    source=img,
                    stream=False,
                    verbose=False,
                    conf=CONF_THRESHOLD,
                    device=device_to_use
                )
            
            detections_results = results_list[0] if results_list else None
            
            if RESULT_QUEUE.full():
                RESULT_QUEUE.get_nowait()
            RESULT_QUEUE.put((detections_results, img)) 
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Prediction Error in Inference Thread: {e}")
            time.sleep(1) 
            continue

def main():
    global AI_ACTIVE, HUMAN_MODE_ACTIVE, SHOW_DEBUG_WINDOW
    
    full_screen_width, full_screen_height = get_primary_monitor_resolution()
    
    MONITOR_LEFT = (full_screen_width - MONITOR_WIDTH) // 2
    MONITOR_TOP = (full_screen_height - MONITOR_HEIGHT) // 2
    
    tensorrt_conversion_success = convert_to_tensorrt() 
    device_to_use = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path_to_use = ENGINE_PATH if tensorrt_conversion_success else MODEL_PATH
    
    if not os.path.exists(model_path_to_use):
        print(f"Error: Model file not found at {model_path_to_use}")
        sys.exit(1)
        
    try:
        model = YOLO(model_path_to_use, task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Model loaded using: {'TensorRT Engine' if tensorrt_conversion_success else 'PyTorch'}")
    print(f"Device: {device_to_use}")

    capture_thread = threading.Thread(target=capture_thread_func, args=(MONITOR_LEFT, MONITOR_TOP), daemon=True)
    inference_thread = threading.Thread(target=inference_thread_func, args=(model, device_to_use), daemon=True)
    
    capture_thread.start()
    inference_thread.start()
    
    last_toggle_time = time.time() 
    screen_center_x = MONITOR_WIDTH // 2
    screen_center_y = MONITOR_HEIGHT // 2
    window_set_topmost_once = False 
    
    img = np.zeros((MONITOR_HEIGHT, MONITOR_WIDTH, 3), dtype=np.uint8)

    while True:
        detections_results = None
        target_pos_for_visual = None
        
        current_time = time.time()
        
        if keyboard.is_pressed('f2') and (current_time - last_toggle_time > 0.5): 
            AI_ACTIVE = not AI_ACTIVE
            last_toggle_time = current_time
            if not AI_ACTIVE:
                global PREV_TARGET_ON_SCREEN, PREV_TARGET_GLOBAL_POS, PREV_TARGET_TIME
                PREV_TARGET_ON_SCREEN = None 
                PREV_TARGET_GLOBAL_POS = None 
                PREV_TARGET_TIME = None
            print(f"AI Toggled: {'ACTIVE' if AI_ACTIVE else 'INACTIVE'}")
            
        if keyboard.is_pressed('f3') and (current_time - last_toggle_time > 0.5): 
            HUMAN_MODE_ACTIVE = not HUMAN_MODE_ACTIVE
            last_toggle_time = current_time
            print(f"Mode Toggled: {'Human' if HUMAN_MODE_ACTIVE else 'Sense'}")

        try:
            detections_results, img = RESULT_QUEUE.get_nowait()
            
            if AI_ACTIVE and detections_results:
                target_pos_for_visual = aim_at_target(
                    detections_results, 
                    screen_center_x, screen_center_y, 
                    MONITOR_LEFT, MONITOR_TOP
                )
            
        except queue.Empty:
            pass 
        
        if SHOW_DEBUG_WINDOW and img is not None:
            
            try:
                if not window_set_topmost_once and cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                    set_window_always_on_top(DEBUG_WINDOW_NAME)
                    window_set_topmost_once = True
            except Exception:
                pass
                
            if detections_results and AI_ACTIVE: 
                for det in detections_results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2 = det[0:4].astype(int)
                    conf = det[4] 
                    cls = int(det[5]) 

                    if cls == TARGET_CLASS and conf >= CONF_THRESHOLD: 
                        color = (255, 0, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f"Enemy: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            if target_pos_for_visual:
                cv2.circle(img, target_pos_for_visual, 5, (0, 0, 255), -1)
                cv2.line(img, (screen_center_x, screen_center_y), target_pos_for_visual, (0, 255, 0), 1)

            status_text = "ACTIVE" if AI_ACTIVE else "INACTIVE"
            color_status = (0, 255, 0) if AI_ACTIVE else (0, 0, 255)
            mode_text = "Human" if HUMAN_MODE_ACTIVE else "Sense"
            cv2.putText(img, f"AI: {status_text} (Mode: {mode_text})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_status, 2)
            
            cv2.line(img, (screen_center_x - 10, screen_center_y), (screen_center_x + 10, screen_center_y), (255, 255, 255), 1)
            cv2.line(img, (screen_center_x, screen_center_y - 10), (screen_center_x, screen_center_y + 10), (255, 255, 255), 1)

            cv2.imshow(DEBUG_WINDOW_NAME, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if not SHOW_DEBUG_WINDOW and cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyAllWindows()
        
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()
