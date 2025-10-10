import sys
import time
import numpy as np
import cv2
import mss
import keyboard
import random
import math
from threading import Thread
from PyQt5 import QtCore, QtGui, QtWidgets

import win32api
import win32con

ACTIVE = False

def smooth_move_mouse_relative(dX_total, dY_total):
    x_start, y_start = 0, 0
    x_end, y_end = dX_total, dY_total
    
    dist = math.sqrt(dX_total**2 + dY_total**2)
    
    if dist < 1.0: 
        if dist > 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(round(dX_total)), int(round(dY_total)), 0, 0)
        return
    
    base_duration = max(0.04, min(0.20, dist / 500.0))
    duration = base_duration * random.uniform(0.9, 1.1)
    steps = max(8, min(70, int(dist // 3)))
    
    if dist > 20:
        ctrl_factor = min(0.25, dist / 100.0) 
        ctrl_x = dX_total // 2 + random.randint(-int(dist*ctrl_factor*0.1), int(dist*ctrl_factor*0.1)) 
        ctrl_y = dY_total // 2 + random.randint(-int(dist*ctrl_factor*0.1), int(dist*ctrl_factor*0.1))
    else:
        ctrl_x = dX_total // 2 + random.randint(-1, 1)
        ctrl_y = dY_total // 2 + random.randint(-1, 1)
        
    last_x, last_y = 0.0, 0.0 
    
    for i in range(steps + 1):
        t = i / steps
        t_eased = t * t * (3 - 2 * t) 
        
        target_x = (1 - t_eased)**2 * x_start + 2 * (1 - t_eased) * t_eased * ctrl_x + t_eased**2 * x_end
        target_y = (1 - t_eased)**2 * y_start + 2 * (1 - t_eased) * t_eased * ctrl_y + t_eased**2 * y_end
        
        if i > 0 and i < steps:
            noise_factor = min(1.5, dist / 100.0) * 0.5 
            target_x += random.uniform(-noise_factor, noise_factor)
            target_y += random.uniform(-noise_factor, noise_factor)
            
        dX_step = int(round(target_x) - round(last_x))
        dY_step = int(round(target_y) - round(last_y))
        
        if dX_step != 0 or dY_step != 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dX_step, dY_step, 0, 0)
        
        last_x = target_x 
        last_y = target_y
        
        time.sleep(duration / steps) 

def move_mouse_relative(dX, dY):
    smooth_move_mouse_relative(dX, dY)


def toggle_active(e):
    global ACTIVE
    ACTIVE = not ACTIVE
    print(f"에임봇 상태: {'활성화됨' if ACTIVE else '비활성화됨'}")

keyboard.on_press_key("f", toggle_active)

class CustomSignals(QtCore.QObject):
    update_signal = QtCore.pyqtSignal(np.ndarray, object, object, str)
    smoothing_changed = QtCore.pyqtSignal(float)

class DetectionViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("에임봇 감지 화면 (F키 토글)")
        self.setGeometry(100, 100, 400, 520)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E; 
                color: #F0F0F0;
                font-family: 'Malgun Gothic', 'Inter';
            }
            QLabel#status_label {
                font-size: 14pt; 
                font-weight: bold; 
                color: #4CAF50;
                padding: 8px;
                border: 1px solid #555555;
                border-radius: 8px;
                background-color: #3A3A3A;
            }
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px 0 5px;
                color: #90CAF9;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #3A3A3A;
                width: 18px;
                margin: -3px 0;
                border-radius: 8px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3A3A3A;
                margin: 2px 0;
                border-radius: 4px;
            }
        """)
        
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        
        self.status_label = QtWidgets.QLabel("비활성화 상태 (F키로 전환)", self)
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # UI for Speed Control (Smoothing Factor)
        control_group = QtWidgets.QGroupBox("에임 속도 조절 (Smoothing)")
        control_layout = QtWidgets.QVBoxLayout()

        self.smoothing_label = QtWidgets.QLabel("현재 속도: 0.50 (표준)")
        self.smoothing_label.setAlignment(QtCore.Qt.AlignCenter)
        control_layout.addWidget(self.smoothing_label)

        self.smoothing_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.smoothing_slider.setRange(10, 100)
        self.smoothing_slider.setValue(50)
        self.smoothing_slider.setTickInterval(10)
        self.smoothing_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        
        self.smoothing_slider.valueChanged.connect(self.update_smoothing_value)
        control_layout.addWidget(self.smoothing_slider)
        
        control_group.setLayout(control_layout)
        
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(control_group)
        self.setLayout(main_layout)
        
        self.show()

    def set_detector_signals(self, detector_signals):
        self.detector_signals = detector_signals
        
    @QtCore.pyqtSlot(int)
    def update_smoothing_value(self, value):
        new_smoothing = value / 100.0
        
        if new_smoothing < 0.3:
            speed_desc = "느림"
        elif new_smoothing < 0.7:
            speed_desc = "표준"
        else:
            speed_desc = "빠름"
            
        self.smoothing_label.setText(f"현재 속도: {new_smoothing:.2f} ({speed_desc})")
        
        if hasattr(self, 'detector_signals'):
            self.detector_signals.smoothing_changed.emit(new_smoothing)
    
    def update_image(self, frame, detected_rect=None, target_point=None, target_ratio_str="N/A"):
        display_frame = frame.copy()
        if detected_rect:
            x, y, w, h = detected_rect
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            if target_point:
                target_x, target_y = target_point
                cv2.circle(display_frame, (target_x, target_y), 5, (0, 0, 255), -1)

        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(display_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        qt_image = convert_to_qt_format.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        
        status_text = f"에임봇 상태: {'🟢 활성화됨' if ACTIVE else '🔴 비활성화됨'}"
        status_text += f" | 타겟: {target_ratio_str}"
        self.status_label.setText(status_text)


class Detector(Thread):
    def __init__(self, viewer):
        super().__init__(daemon=True)
        self.viewer = viewer
        self.signals = CustomSignals() 
        self.signals.update_signal.connect(self.viewer.update_image)
        self.signals.smoothing_changed.connect(self.update_smoothing_factor)
        
        with mss.mss() as sct:
            MON = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            screen_width, screen_height = MON["width"], MON["height"]
            capture_size = 350
            center_x, center_y = screen_width // 2, screen_height // 2
            self.capture_area = {
                "left": center_x - capture_size // 2,
                "top": center_y - capture_size // 2,
                "width": capture_size,
                "height": capture_size
            }
            self.capture_center_x = capture_size // 2
            self.capture_center_y = capture_size // 2
            
        self.low1, self.hi1 = np.array([0,150,150]), np.array([10,255,255])
        self.low2, self.hi2 = np.array([170,150,150]), np.array([180,255,255])
        
        self.target_fps = 400
        self.target_frame_time = 1.0 / self.target_fps
        self.deadzone = 7
        self.smoothing_factor = 0.5 

        self.current_target_ratio = 0.5
        self.current_ratio_description = "중앙 지점 (2분의 1)"
    
    # @QtCore.pyqtSlot(float) 데코레이터를 제거하여 TypeError를 해결했습니다.
    def update_smoothing_factor(self, new_value):
        self.smoothing_factor = new_value

    def run(self):
        time.sleep(1)
        
        with mss.mss() as sct:
            while True:
                loop_start_time = time.time()
                
                img = sct.grab(self.capture_area) 
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected_rect = None
                target_point = None

                if ACTIVE:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(hsv, self.low1, self.hi1)
                    mask2 = cv2.inRange(hsv, self.low2, self.hi2)
                    mask = mask1 | mask2
                    
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=4)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kern, iterations=2)
                    
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = [c for c in cnts if cv2.contourArea(c) > 2000]
                    
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        x,y,ww,hh = cv2.boundingRect(c)
                        detected_rect = (x, y, ww, hh)
                        
                        target_center_x_in_capture = x + ww // 2
                        target_center_y_in_capture = y + int(hh * self.current_target_ratio)

                        target_point = (target_center_x_in_capture, target_center_y_in_capture)
                        
                        dX_err = target_center_x_in_capture - self.capture_center_x
                        dY_err = target_center_y_in_capture - self.capture_center_y
                        
                        if abs(dX_err) < self.deadzone and abs(dY_err) < self.deadzone:
                            dX, dY = 0, 0
                        else:
                            dX = dX_err * self.smoothing_factor
                            dY = dY_err * self.smoothing_factor

                            max_delta = 50 
                            dX = max(-max_delta, min(max_delta, dX))
                            dY = max(-max_delta, min(max_delta, dY))

                            move_mouse_relative(dX, dY)
                    
                self.signals.update_signal.emit(frame, detected_rect, target_point, self.current_ratio_description)
                
                loop_end_time = time.time()
                elapsed_time = loop_end_time - loop_start_time
                sleep_time = self.target_frame_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv) 
    
    viewer = DetectionViewer() 
    
    detector_thread = Detector(viewer)
    viewer.set_detector_signals(detector_thread.signals)
    detector_thread.start()
    
    print("F 키를 눌러 에임봇을 활성화/비활성화할 수 있습니다.")
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass
