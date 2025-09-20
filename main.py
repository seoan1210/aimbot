import cv2
import numpy as np
from mss import mss
import time
from win32 import win32api
import win32con


MONITOR_WIDTH, MONITOR_HEIGHT = map(int, input("your monitors (ex: 1920 1080): ").split())
BOX_SIZE = 400


LOWER_COLOR_RANGE1 = np.array([0, 100, 100])
UPPER_COLOR_RANGE1 = np.array([10, 255, 255])
LOWER_COLOR_RANGE2 = np.array([160, 100, 100])
UPPER_COLOR_RANGE2 = np.array([179, 255, 255])



sct = mss()

def grab_screen():
    monitor_info = sct.monitors[1]
    box = {
        'top': monitor_info['top'] + int((MONITOR_HEIGHT / 2) - (BOX_SIZE / 2)),
        'left': monitor_info['left'] + int((MONITOR_WIDTH / 2) - (BOX_SIZE / 2)),
        'width': BOX_SIZE,
        'height': BOX_SIZE,
    }
    sct_img = sct.grab(box)
    return np.array(sct_img)

def move_mouse(x_offset, y_offset, human_mode=True):
    if human_mode:
        smoothness = 10
        x_move = int(x_offset / smoothness)
        y_move = int(y_offset / smoothness)
        
        if x_move == 0 and x_offset != 0:
            x_move = 1 if x_offset > 0 else -1
        if y_move == 0 and y_offset != 0:
            y_move = 1 if y_offset > 0 else -1
        
        threshold = 1
        if abs(x_offset) < threshold and abs(y_offset) < threshold:
            return
            
    else:
        x_move = x_offset
        y_move = y_offset

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_move, y_move, 0, 0)

def process_image(image, human_mode=True):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    red_mask = cv2.inRange(hsv, LOWER_COLOR_RANGE1, UPPER_COLOR_RANGE1)
    red_mask |= cv2.inRange(hsv, LOWER_COLOR_RANGE2, UPPER_COLOR_RANGE2)
    
    kernel = np.ones((5, 5), np.uint8)
    processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return

    health_bar_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        if aspect_ratio > 3.0 and h < 20 and y < BOX_SIZE / 2:
            if health_bar_contour is None or area > cv2.contourArea(health_bar_contour):
                health_bar_contour = c
    
    if health_bar_contour is not None:
        moment = cv2.moments(health_bar_contour)
        if moment["m00"] != 0:
            center_x = int(moment["m10"] / moment["m00"])
            center_y = int(moment["m01"] / moment["m00"])

            center_x_offset = int(center_x - (BOX_SIZE / 2)) + 30
            center_y_offset = int(center_y - (BOX_SIZE / 2)) + 60

            move_mouse(center_x_offset, center_y_offset, human_mode=human_mode)

            x, y, w, h = cv2.boundingRect(health_bar_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)


def main():
    print("Aimvbot is running... Press 'q' to quit.")
    
    human_aim_mode = True
    
    while True:
        loop_start_time = time.time()
        
        captured_image = grab_screen()
        process_image(captured_image, human_mode=human_aim_mode)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
