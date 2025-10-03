import cv2
import numpy as np
from mss import mss
import time
from win32 import win32api
import win32con
import threading
import tkinter as tk
from tkinter import ttk

def get_monitor_resolution():
    try:
        w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        return w, h
    except Exception as e:
        print(f"ERROR: {e}")
        return 1920, 1080 

MONITOR_WIDTH, MONITOR_HEIGHT = get_monitor_resolution()
BOX_SIZE = 1000

LOWER_COLOR_RANGE1 = np.array([0, 100, 100])
UPPER_COLOR_RANGE1 = np.array([10, 255, 255])
LOWER_COLOR_RANGE2 = np.array([160, 100, 100])
UPPER_COLOR_RANGE2 = np.array([179, 255, 255])


def grab_screen(sct_instance):
    try:
        monitor_info = sct_instance.monitors[1] 
    except IndexError:
        monitor_info = sct_instance.monitors[0]
        
    box = {
        'top': monitor_info['top'] + int((MONITOR_HEIGHT / 2) - (BOX_SIZE / 2)),
        'left': monitor_info['left'] + int((MONITOR_WIDTH / 2) - (BOX_SIZE / 2)),
        'width': BOX_SIZE,
        'height': BOX_SIZE,
    }
    sct_img = sct_instance.grab(box)
    return np.array(sct_img)

def move_mouse(x_offset, y_offset, human_mode=True):
    
    if human_mode:
        smoothness = 35
        x_move = int(x_offset / smoothness)
        y_move = int(y_offset / smoothness)
        
        if x_move == 0 and x_offset != 0:
            x_move = 1 if x_offset > 0 else -1
        if y_move == 0 and y_offset != 0:
            y_move = 1 if y_offset > 0 else -1
            
        threshold = 3
        if abs(x_offset) < threshold and abs(y_offset) < threshold:
            return
    else:
        sensitivity = 0.4

        x_move = int(x_offset * sensitivity)
        y_move = int(y_offset * sensitivity)

        if x_move == 0 and x_offset != 0:
            x_move = 1 if x_offset > 0 else -1
        if y_move == 0 and y_offset != 0:
            y_move = 1 if y_offset > 0 else -1
            
        threshold = 2
        if abs(x_offset) <= threshold and abs(y_offset) <= threshold:
            return

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_move, y_move, 0, 0)


class AimbotApp:
    def __init__(self, master):
        self.master = master
        master.title("Aimbot Controller 🤖")
        
        self.is_running = False
        self.aimbot_thread = None
        
        self.human_aim_mode = tk.BooleanVar(value=True) 
        self.status_text = tk.StringVar(value="Status: Ready")
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 10, 'bold'), padding=5)
        style.configure('TLabel', font=('Helvetica', 10), padding=5)

        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.start_button = ttk.Button(main_frame, text="Start Aimbot", command=self.toggle_aimbot, style='TButton')
        self.start_button.grid(row=0, column=0, columnspan=2, pady=10, sticky='ew')
        
        mode_frame = ttk.LabelFrame(main_frame, text="Aim Mode", padding="5")
        mode_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        
        ttk.Radiobutton(mode_frame, text="Human Mode (Smooth)", variable=self.human_aim_mode, value=True, command=self.update_status).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Fast Mode (Precise)", variable=self.human_aim_mode, value=False, command=self.update_status).pack(anchor='w')
        
        self.status_label = ttk.Label(main_frame, textvariable=self.status_text, style='TLabel')
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10, sticky='ew')
        
        self.close_button = ttk.Button(main_frame, text="Exit", command=self.close_app, style='TButton')
        self.close_button.grid(row=3, column=0, columnspan=2, pady=10, sticky='ew')

        res_label = ttk.Label(main_frame, text=f"Resolution: {MONITOR_WIDTH}x{MONITOR_HEIGHT} | Box: {BOX_SIZE}x{BOX_SIZE}", font=('Helvetica', 8))
        res_label.grid(row=4, column=0, columnspan=2, sticky='ew')

        self.update_status()
        
    def stop_aimbot(self):
        self.is_running = False
        self.update_status()

    def close_app(self):
        self.stop_aimbot()
        self.master.destroy()

    def update_status(self):
        mode = "Human Mode" if self.human_aim_mode.get() else "Fast Mode"
        if self.is_running:
            status = f"Status: RUNNING! ({mode})"
            self.status_label.config(foreground="green")
            self.start_button.config(text="Stop Aimbot")
        else:
            status = f"Status: Stopped. Current Mode: {mode}"
            self.status_label.config(foreground="red")
            self.start_button.config(text="Start Aimbot")
        self.status_text.set(status)

    def toggle_aimbot(self):
        if self.is_running:
            self.stop_aimbot()
        else:
            self.start_aimbot()

    def start_aimbot(self):
        if self.aimbot_thread is None or not self.aimbot_thread.is_alive():
            self.is_running = True
            self.aimbot_thread = threading.Thread(target=self.aimbot_loop, daemon=True)
            self.aimbot_thread.start()
            self.update_status()
            
    def process_image(self, image, human_mode=True):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.inRange(hsv, LOWER_COLOR_RANGE1, UPPER_COLOR_RANGE1)
        red_mask |= cv2.inRange(hsv, LOWER_COLOR_RANGE2, UPPER_COLOR_RANGE2)
        
        kernel = np.ones((3, 3), np.uint8)
        processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

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

    def aimbot_loop(self):
        print("Aimbot is running...")
        
        try:
            thread_sct = mss() 
        except Exception as e:
            print(f"ERROR creating mss instance in thread: {e}")
            self.is_running = False
            self.update_status()
            return

        while self.is_running:
            
            captured_image = grab_screen(thread_sct) 
            
            current_human_mode = self.human_aim_mode.get() 
            self.process_image(captured_image, human_mode=current_human_mode)
            
            time.sleep(0.001) 
            
        print("Aimbot stopped.")
        cv2.destroyAllWindows()


def main_gui():
    root = tk.Tk()
    app = AimbotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app) 
    root.mainloop()

if __name__ == "__main__":
    main_gui()
