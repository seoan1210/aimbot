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

MONITOR_WIDTH = 320
MONITOR_HEIGHT = 320

MODEL_PATH = 'overwatch.pt'
ENGINE_PATH = 'overwatch_tensorrt.engine'

CONF_THRESHOLD = 0.5
TARGET_CLASS = 0
SMOOTHING = 1.2
MAX_DISTANCE = 450

ACCUMULATED_X_ERROR = 0.0
ACCUMULATED_Y_ERROR = 0.0

DEBUG_WINDOW_NAME = "Overwatch AI | F2: Toggle AI | F3: Toggle Mode"
HUMAN_MODE_ACTIVE = False

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
        sensitivity = 0.3
        
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
        return True
    
    try:
        model = YOLO(MODEL_PATH)
        success_path = model.export(
            format='engine', 
            half=True, 
            imgsz=[MONITOR_HEIGHT, MONITOR_WIDTH], 
            device='0',
            simplify=True
        )
        
        if success_path and os.path.exists(success_path):
            if success_path != ENGINE_PATH: 
                if os.path.exists(ENGINE_PATH): 
                    os.remove(ENGINE_PATH)
                os.rename(success_path, ENGINE_PATH)
            return True
        return False
            
    except Exception: 
        return False

def capture_screen(monitor_left, monitor_top):
    with mss() as sct:
        monitor_config = {
            "top": monitor_top, 
            "left": monitor_left, 
            "width": MONITOR_WIDTH, 
            "height": MONITOR_HEIGHT
        }
        screenshot = np.array(sct.grab(monitor_config))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR) 

def aim_at_target(detections_data, screen_center_x, screen_center_y, monitor_left, monitor_top, prev_target=None):
    if not detections_data or detections_data.boxes.data.numel() == 0:
        return None
    
    detections = detections_data.boxes.data.cpu().numpy()
    
    valid_detections = detections[(detections[:, 5] == TARGET_CLASS) & (detections[:, 4] >= CONF_THRESHOLD)]
    
    if valid_detections.size == 0:
        return None
    
    x1, y1, x2, y2 = valid_detections[:, 0:4].T
    
    target_center_x = ((x1 + x2) / 2).astype(int)
    target_center_y = (y1 + (y2 - y1) / 6).astype(int)

    distances = np.sqrt((target_center_x - screen_center_x) ** 2 + (target_center_y - screen_center_y) ** 2)
    
    close_targets_indices = np.where(distances <= MAX_DISTANCE)[0]
    
    if close_targets_indices.size == 0:
        return None
    
    best_target_index = close_targets_indices[np.argmin(distances[close_targets_indices])]
    
    target_x = target_center_x[best_target_index]
    target_y = target_center_y[best_target_index]
    
    if prev_target and SMOOTHING > 1.0:
        prev_x, prev_y = prev_target
        target_x = int(prev_x + (target_x - prev_x) / SMOOTHING)
        target_y = int(prev_y + (target_y - prev_y) / SMOOTHING)
    else:
        target_x, target_y = int(target_x), int(target_y)

    current_mouse_x, current_mouse_y = win32api.GetCursorPos()
    
    global_target_x = monitor_left + target_x
    global_target_y = monitor_top + target_y

    delta_x = global_target_x - current_mouse_x
    delta_y = global_target_y - current_mouse_y
    
    move_mouse(delta_x, delta_y, human_mode=HUMAN_MODE_ACTIVE)
    
    return (target_x, target_y)

def set_window_always_on_top(window_title):
    try:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    except Exception:
        pass 

def main():
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

    global HUMAN_MODE_ACTIVE
    ai_active = False 
    last_toggle_time = time.time() 
    screen_center_x = MONITOR_WIDTH // 2
    screen_center_y = MONITOR_HEIGHT // 2
    
    prev_target_on_screen = None 
    window_set_topmost_once = False 

    with torch.no_grad():
        while True:
            img = capture_screen(MONITOR_LEFT, MONITOR_TOP)
            
            try:
                cv2.imshow(DEBUG_WINDOW_NAME, img)
                if not window_set_topmost_once and cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                    set_window_always_on_top(DEBUG_WINDOW_NAME)
                    window_set_topmost_once = True
            except Exception:
                pass
                
            detections_results = None 
            target_pos_for_visual = None 

            if ai_active:
                try:
                    results_list = model.predict(
                        source=img,
                        stream=False,
                        verbose=False,
                        conf=CONF_THRESHOLD,
                        device=device_to_use
                    )
                    
                    if results_list: 
                        detections_results = results_list[0] 
                        
                        target_pos_for_visual = aim_at_target(
                            detections_results, 
                            screen_center_x, screen_center_y, 
                            MONITOR_LEFT, MONITOR_TOP,
                            prev_target_on_screen
                        )
                        
                        prev_target_on_screen = target_pos_for_visual
                    else:
                        prev_target_on_screen = None 
                except Exception as e:
                    ai_active = False
                    prev_target_on_screen = None
                    print(f"Prediction or Aiming Error: {e}")

            else:
                prev_target_on_screen = None 

            if window_set_topmost_once:
                if detections_results and ai_active: 
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

                status_text = "ACTIVE" if ai_active else "INACTIVE"
                color_status = (0, 255, 0) if ai_active else (0, 0, 255)
                mode_text = "Human" if HUMAN_MODE_ACTIVE else "Sense"
                cv2.putText(img, f"AI: {status_text} (Mode: {mode_text})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_status, 2)
                
                cv2.line(img, (screen_center_x - 10, screen_center_y), (screen_center_x + 10, screen_center_y), (255, 255, 255), 1)
                cv2.line(img, (screen_center_x, screen_center_y - 10), (screen_center_x, screen_center_y + 10), (255, 255, 255), 1)

                cv2.imshow(DEBUG_WINDOW_NAME, img)
                
            if keyboard.is_pressed('f2') and (time.time() - last_toggle_time > 0.5): 
                ai_active = not ai_active
                last_toggle_time = time.time()
                if not ai_active:
                    prev_target_on_screen = None 
                
            if keyboard.is_pressed('f3') and (time.time() - last_toggle_time > 0.5): 
                HUMAN_MODE_ACTIVE = not HUMAN_MODE_ACTIVE
                last_toggle_time = time.time()
                
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
