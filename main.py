import sys
import time
import subprocess
from threading import Thread
import random
import os
import math

import numpy as np
import cv2
import mss
import keyboard
import win32api
import win32con
import torch
from ultralytics import YOLO # 🚨 AI 모델 라이브러리 추가
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

# --- AI 및 시스템 설정 상수 ---
MODEL_PATH = './v2.pt'  # 🚨 네가 학습시킨 YOLO 모델 파일 경로
ACTIVE = False
CROP_WIDTH = 1000
CROP_HEIGHT = 350 
DEFAULT_SMOOTHING_FACTOR = 0.07
DEADZONE = 10 
RANDOM_SMOOTHING_ENABLED = False 
RANDOM_INTERVAL_SECONDS = 5.0
CURRENT_SMOOTHING_FACTOR = DEFAULT_SMOOTHING_FACTOR 
LANGUAGE = "ko" 

# YOLOv8은 클래스 ID 0부터 시작하므로, 타겟 클래스 ID를 설정
TARGET_CLASSES = [0] # 예시: 모델에서 '사람' 또는 '적'의 클래스 ID가 0인 경우

# --- 언어 설정 (변경 없음) ---
LANGUAGES = {
    "ko": {
        "title": "에임봇 컨트롤 패널 (AI)",
        "lang_select": "언어 선택",
        "status": "상태",
        "active": "활성화됨",
        "inactive": "비활성화됨",
        "toggle_key": "토글 키",
        "target_point": "타겟 지점",
        "smoothing_factor": "스무딩 계수",
        "ratio_group": "조준 지점 조절",
        "ratio_selected": "선택됨",
        "ratio_head": "머리 (0.20)",
        "ratio_neck": "목 (0.35)",
        "ratio_body": "몸통 중앙 (0.50)",
        "ratio_pelvis": "골반 (0.65)",
        "smoothing_group": "마우스 스무딩 (WMA Factor)",
        "smoothing_current": "현재 WMA Factor",
        "smoothing_max": "최대",
        "random_smoothing": "랜덤 스무딩 ({s}초마다, 최대 0.35)",
        "random": "랜덤",
        "initializing": "AI 초기화 중...",
        "enemy": "적",
    },
    # EN 설정은 생략...
    "en": {
        "title": "Aim Bot Control Panel (AI)",
        "lang_select": "Language",
        "status": "Status",
        "active": "Active",
        "inactive": "Inactive",
        "toggle_key": "Toggle Key",
        "target_point": "Target Point",
        "smoothing_factor": "Smoothing Factor",
        "ratio_group": "Target Point Adjustment",
        "ratio_selected": "Selected",
        "ratio_head": "Head (0.20)",
        "ratio_neck": "Neck (0.35)",
        "ratio_body": "Body Center (0.50)",
        "ratio_pelvis": "Pelvis (0.65)",
        "smoothing_group": "Mouse Smoothing (WMA Factor)",
        "smoothing_current": "Current WMA Factor",
        "smoothing_max": "Max",
        "random_smoothing": "Random Smoothing (every {s}s, max 0.35)",
        "random": "Random",
        "initializing": "Initializing...",
        "enemy": "Enemy",
    }
}

# --- 마우스 이동 함수 (변경 없음) ---
def move_mouse_relative_linear(dX, dY):
    move_x = int(round(dX))
    move_y = int(round(dY))
    
    if move_x != 0 or move_y != 0:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

def move_mouse_relative(dX, dY):
    move_mouse_relative_linear(dX, dY)

# --- 토글 키 함수 (변경 없음) ---
def toggle_active(e):
    global ACTIVE
    ACTIVE = not ACTIVE

keyboard.on_press_key("f", toggle_active)

# --- 오버레이 창 클래스 (변경 없음) ---
class OverlayWindow(QtWidgets.QWidget):
    
    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | 
            QtCore.Qt.WindowStaysOnTopHint | 
            QtCore.Qt.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        self.setGeometry(0, 0, screen_w, screen_h)
        
        self.list_of_enemies = []
        self.target_point = None

    @QtCore.pyqtSlot(object, object)
    def update_overlay(self, list_of_enemies, target_point):
        self.list_of_enemies = list_of_enemies
        self.target_point = target_point
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        
        current_lang = LANGUAGES.get(LANGUAGE, LANGUAGES["ko"])
        
        if self.list_of_enemies:
            for enemy_data in self.list_of_enemies:
                enemy_id, x, y, w, h, is_target = enemy_data
                
                if ACTIVE and is_target:
                    color = QtGui.QColor(0, 191, 255, 220) 
                else:
                    color = QtGui.QColor(50, 205, 50, 180) 

                painter.setPen(QtGui.QPen(color, 4)) 
                painter.drawRect(x, y, w, h)

                painter.setFont(QtGui.QFont("Malgun Gothic", 10, QtGui.QFont.Bold))
                painter.setPen(QtGui.QPen(color))
                painter.drawText(x, y - 15, f"{current_lang['enemy']}: {enemy_id}")
        
        if ACTIVE and self.target_point:
            tx, ty = self.target_point
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0))) 
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(tx - 3, ty - 3, 6, 6) 

        painter.end()


# --- 시그널 클래스 및 UI 클래스 (변경 없음) ---
class CustomSignals(QtCore.QObject):
    update_status_signal = QtCore.pyqtSignal(str, float, bool) 
    update_overlay_signal = QtCore.pyqtSignal(object, object) 
    ratio_changed = QtCore.pyqtSignal(float, str)
    smoothing_factor_set = QtCore.pyqtSignal(float)
    update_smoothing_gui = QtCore.pyqtSignal(float)
    language_changed = QtCore.pyqtSignal()

class DetectionViewer(QtWidgets.QWidget):
    
    TARGET_RATIOS = {
        "ko": {
            "머리 (0.20)": 0.20,
            "목 (0.35)": 0.35,
            "몸통 중앙 (0.50)": 0.50,
            "골반 (0.65)": 0.65,
        },
        "en": {
            "Head (0.20)": 0.20,
            "Neck (0.35)": 0.35,
            "Body Center (0.50)": 0.50,
            "Pelvis (0.65)": 0.65,
        }
    }
    
    def __init__(self):
        super().__init__()
        self.current_lang = LANGUAGES[LANGUAGE]
        self.setGeometry(100, 100, 450, 500) 
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setStyleSheet(self._get_style_sheet())
        
        self._setup_ui()
        self.update_texts() 
        self.show()
        
    def _get_style_sheet(self):
        # 스타일 시트 (생략)
        return """
            QWidget { background-color: #1A1A1A; color: #E0E0E0; font-family: 'Segoe UI', 'Malgun Gothic', 'Inter'; }
            QLabel { font-size: 10pt; color: #B0B0B0; }
            QLabel#status_label { 
                font-size: 12pt; font-weight: bold; 
                padding: 10px; border-radius: 10px; 
                background-color: #2A2A2A; 
                border: 1px solid #444444; 
                min-height: 80px;
                qproperty-alignment: AlignCenter;
            }
            QGroupBox { 
                border: 2px solid #444444; 
                border-radius: 10px; margin-top: 15px; 
                padding-top: 10px; 
                background-color: #222222;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; left: 15px; 
                padding: 0 5px 0 5px; 
                color: #88CCFF; 
                font-size: 11pt; font-weight: bold; 
            }
            QComboBox { 
                background-color: #333333; color: #E0E0E0; 
                border: 1px solid #555555; padding: 8px; 
                border-radius: 5px; font-size: 10pt; 
                selection-background-color: #007ACC;
            }
            QSlider::groove:horizontal { 
                border: 1px solid #555555; height: 10px; 
                background: #333333; margin: 2px 0; border-radius: 5px; 
            }
            QSlider::handle:horizontal { 
                background: #00AACC; border: 1px solid #00AACC; 
                width: 20px; margin: -5px 0; border-radius: 10px; 
            }
            QSlider::handle:horizontal:hover { background: #00BFFF; }
            QCheckBox { spacing: 8px; color: #B0B0B0; font-size: 10pt; }
            QCheckBox::indicator { width: 18px; height: 18px; }
            QCheckBox::indicator:unchecked { border: 1px solid #777777; background-color: #444444; border-radius: 4px; }
            QCheckBox::indicator:checked { border: 1px solid #007ACC; background-color: #00AACC; border-radius: 4px; }
        """

    def _setup_ui(self):
        # UI 설정 (생략)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        lang_layout = QtWidgets.QHBoxLayout()
        self.lang_label = QtWidgets.QLabel()
        self.lang_combo_box = QtWidgets.QComboBox(self)
        self.lang_combo_box.addItem("한국어 (Korean)", "ko")
        self.lang_combo_box.addItem("English", "en")
        self.lang_combo_box.setCurrentIndex(0 if LANGUAGE == "ko" else 1)
        self.lang_combo_box.currentIndexChanged.connect(self.change_language)
        lang_layout.addWidget(self.lang_label)
        lang_layout.addWidget(self.lang_combo_box)
        main_layout.addLayout(lang_layout)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True) 
        main_layout.addWidget(self.status_label)

        self.ratio_group = QtWidgets.QGroupBox()
        ratio_layout = QtWidgets.QVBoxLayout(self.ratio_group)
        ratio_layout.setSpacing(10)
        
        self.ratio_combo_box = QtWidgets.QComboBox(self)
        self.ratio_combo_box.currentIndexChanged.connect(self.update_ratio_value)
        ratio_layout.addWidget(self.ratio_combo_box)
        
        self.ratio_label = QtWidgets.QLabel()
        self.ratio_label.setAlignment(QtCore.Qt.AlignCenter)
        ratio_layout.addWidget(self.ratio_label)
        
        main_layout.addWidget(self.ratio_group)
        
        self.smoothing_group = QtWidgets.QGroupBox()
        smoothing_layout = QtWidgets.QVBoxLayout(self.smoothing_group)
        smoothing_layout.setSpacing(10)
        
        self.random_checkbox = QtWidgets.QCheckBox()
        self.random_checkbox.setChecked(RANDOM_SMOOTHING_ENABLED)
        self.random_checkbox.stateChanged.connect(self.toggle_random_smoothing)
        smoothing_layout.addWidget(self.random_checkbox)
        
        self.smoothing_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.smoothing_slider.setRange(1, 60) 
        initial_value = int(DEFAULT_SMOOTHING_FACTOR * 100) 
        self.smoothing_slider.setValue(initial_value)
        self.smoothing_slider.valueChanged.connect(self.update_smoothing_value)
        self.smoothing_slider.setEnabled(not RANDOM_SMOOTHING_ENABLED) 
        
        self.smoothing_label = QtWidgets.QLabel()
        self.smoothing_label.setAlignment(QtCore.Qt.AlignCenter)
        
        smoothing_layout.addWidget(self.smoothing_label)
        smoothing_layout.addWidget(self.smoothing_slider)
        
        main_layout.addWidget(self.smoothing_group) 
    
    # 나머지 UI 메소드 (update_texts, change_language 등)는 변경 없음
    def update_texts(self):
        self.current_lang = LANGUAGES[LANGUAGE]
        s = self.current_lang 

        self.setWindowTitle(s["title"])
        self.lang_label.setText(s["lang_select"] + ":")
        
        self.ratio_combo_box.blockSignals(True)
        current_ratio_index = self.ratio_combo_box.currentIndex()
        self.ratio_combo_box.clear()
        
        ratio_keys = list(self.TARGET_RATIOS[LANGUAGE].keys())
        for key in ratio_keys:
            self.ratio_combo_box.addItem(key)
        
        if current_ratio_index != -1:
            if current_ratio_index < len(ratio_keys):
                self.ratio_combo_box.setCurrentIndex(current_ratio_index)
            else:
                self.ratio_combo_box.setCurrentIndex(2 if LANGUAGE == "ko" else 2)
        else:
            self.ratio_combo_box.setCurrentIndex(2 if LANGUAGE == "ko" else 2)
            
        self.ratio_combo_box.blockSignals(False)
        self.update_ratio_value(self.ratio_combo_box.currentIndex())
        
        self.ratio_group.setTitle(s["ratio_group"])
        self.smoothing_group.setTitle(s["smoothing_group"])
        self.random_checkbox.setText(s["random_smoothing"].format(s=int(RANDOM_INTERVAL_SECONDS)))
        self.update_smoothing_label(CURRENT_SMOOTHING_FACTOR)

        self.update_status_info(self.ratio_label.text().split(': ')[-1], CURRENT_SMOOTHING_FACTOR, ACTIVE)


    def change_language(self, index):
        global LANGUAGE
        new_lang = self.lang_combo_box.itemData(index)
        if new_lang != LANGUAGE:
            LANGUAGE = new_lang
            self.update_texts()
            if hasattr(self, 'detector_signals'):
                self.detector_signals.language_changed.emit()


    def set_detector_signals(self, detector_signals):
        self.detector_signals = detector_signals
        self.detector_signals.update_smoothing_gui.connect(self.update_smoothing_gui_from_detector) 
        self.detector_signals.update_status_signal.connect(self.update_status_info) 
        self.detector_signals.language_changed.connect(self.update_texts) 
        
    @QtCore.pyqtSlot(int)
    def update_ratio_value(self, index):
        selected_key = self.ratio_combo_box.itemText(index)
        new_ratio = self.TARGET_RATIOS[LANGUAGE][selected_key]
        
        self.ratio_label.setText(f"{self.current_lang['ratio_selected']}: {selected_key}")
        
        if hasattr(self, 'detector_signals'):
            self.detector_signals.ratio_changed.emit(new_ratio, selected_key)

    def update_smoothing_label(self, new_smoothing):
        s = self.current_lang
        
        if RANDOM_SMOOTHING_ENABLED:
            text = f"{s['smoothing_current']} ({s['random']}): {new_smoothing:.2f}"
        else:
            text = f"{s['smoothing_current']}: {new_smoothing:.2f} ({s['smoothing_max']} 0.60)"
            
        self.smoothing_label.setText(text)

    @QtCore.pyqtSlot(int)
    def update_smoothing_value(self, value):
        global CURRENT_SMOOTHING_FACTOR
        new_smoothing = value / 100.0
        CURRENT_SMOOTHING_FACTOR = new_smoothing
        self.update_smoothing_label(new_smoothing)
        
        if hasattr(self, 'detector_signals'):
            self.detector_signals.smoothing_factor_set.emit(new_smoothing)
            
    @QtCore.pyqtSlot(float)
    def update_smoothing_gui_from_detector(self, new_smoothing):
        global CURRENT_SMOOTHING_FACTOR
        CURRENT_SMOOTHING_FACTOR = new_smoothing
        
        if RANDOM_SMOOTHING_ENABLED:
            self.smoothing_slider.setValue(int(new_smoothing * 100))
            self.update_smoothing_label(new_smoothing)

    @QtCore.pyqtSlot(int)
    def toggle_random_smoothing(self, state):
        global RANDOM_SMOOTHING_ENABLED, CURRENT_SMOOTHING_FACTOR
        is_checked = state == QtCore.Qt.Checked
        RANDOM_SMOOTHING_ENABLED = is_checked
        self.smoothing_slider.setEnabled(not is_checked)
        
        current_value = self.smoothing_slider.value() / 100.0
        
        if is_checked:
            if hasattr(self, 'detector_signals'):
                self.detector_signals.smoothing_factor_set.emit(-1.0) 
            self.update_smoothing_label(CURRENT_SMOOTHING_FACTOR)
        else:
            if hasattr(self, 'detector_signals'):
                self.detector_signals.smoothing_factor_set.emit(current_value)
            CURRENT_SMOOTHING_FACTOR = current_value
            self.update_smoothing_label(current_value)


    @QtCore.pyqtSlot(str, float, bool) 
    def update_status_info(self, target_ratio_str, current_smoothing_factor, active_state):
        s = LANGUAGES.get(LANGUAGE, LANGUAGES["ko"])
        
        status_color = "#4CAF50" if active_state else "#FF6347" 
        status_text_active = s['active'] if active_state else s['inactive']
        status_toggle_key = "F"
        
        status_html = (
            f"<p style='color: {status_color}; font-size: 14pt;'><b>{s['status']}: {status_text_active}</b></p>"
            f"<p style='font-size: 10pt;'>{s['toggle_key']}: <span style='color: #00BFFF;'>{status_toggle_key}</span></p>"
            f"<p style='font-size: 10pt;'>{s['target_point']}: <span style='color: #00BFFF;'>{target_ratio_str.split(' (')[0].strip()}</span></p>"
            f"<p style='font-size: 10pt;'>{s['smoothing_factor']}: <span style='color: #00BFFF;'>{current_smoothing_factor:.2f}</span></p>"
        )
        self.status_label.setText(status_html)
        self.status_label.setStyleSheet(f"QLabel#status_label {{ background-color: #2A2A2A; border: 1px solid {status_color}; border-radius: 10px; qproperty-alignment: AlignCenter; padding: 10px; }}")

    def closeEvent(self, event):
        event.accept() 
        sys.exit(0)


# --- 🚨 Detector 클래스 (YOLOv8 기반으로 대폭 수정) 🚨 ---
class Detector(Thread):
    
    def __init__(self, viewer, full_screen_w, full_screen_h, mon_crop):
        super().__init__(daemon=True)
        self.viewer = viewer
        self.signals = CustomSignals() 
        self.signals.update_status_signal.connect(self.viewer.update_status_info) 
        self.signals.language_changed.connect(self.update_current_language_for_overlay)
        
        self.signals.ratio_changed.connect(self.update_target_ratio) 
        self.signals.smoothing_factor_set.connect(self.update_smoothing_factor)
        self.signals.update_smoothing_gui.connect(self.viewer.update_smoothing_gui_from_detector)

        self.full_screen_w = full_screen_w
        self.full_screen_h = full_screen_h
        
        self.MON_CROP = mon_crop
        self.CROP_WIDTH = self.MON_CROP["width"]
        self.CROP_HEIGHT = self.MON_CROP["height"]
        self.CROP_CENTER_X = self.CROP_WIDTH // 2
        self.CROP_CENTER_Y = self.CROP_HEIGHT // 2
        
        # 🚨 YOLOv8 모델 초기화 🚨
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: YOLO model not found at {MODEL_PATH}. Check the path.")
            # 모델 로드 실패 시 강제 종료
            sys.exit(1)
            
        self.model = YOLO(MODEL_PATH)
        
        if torch.cuda.is_available():
            self.model.to("cuda")
            self.half_precision = True
            print("✅ YOLO model loaded to CUDA (GPU).")
        else:
            self.half_precision = False
            print("⚠️ CUDA not available. YOLO model running on CPU.")
            
        self.model.conf = 0.50  # 최소 신뢰도
        self.model.max_det = 10 # 최대 탐지 객체 수

        self.smoothing_factor = DEFAULT_SMOOTHING_FACTOR
        self.DEADZONE = DEADZONE 
        
        self.current_target_ratio = 0.5
        self.current_ratio_description = LANGUAGES[LANGUAGE]["ratio_body"]
        
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        
        self.stop_event = False
        self.random_thread = Thread(target=self.random_smoothing_changer, daemon=True)
        self.random_thread.start()
        
        self.overlay = OverlayWindow(self.full_screen_w, self.full_screen_h) 
        self.signals.update_overlay_signal.connect(self.overlay.update_overlay)
        self.overlay.show()
        
    def update_current_language_for_overlay(self):
        self.overlay.update()

    def update_target_ratio(self, new_ratio, new_desc):
        self.current_target_ratio = new_ratio
        self.current_ratio_description = new_desc

    def update_smoothing_factor(self, new_factor):
        if new_factor == -1.0:
            pass 
        elif not RANDOM_SMOOTHING_ENABLED:
            self.smoothing_factor = new_factor
        
    def random_smoothing_changer(self):
        while not self.stop_event:
            if RANDOM_SMOOTHING_ENABLED:
                new_random_factor = random.randint(10, 35) / 100.0 
                
                self.smoothing_factor = new_random_factor
                self.signals.update_smoothing_gui.emit(new_random_factor)
                
            time.sleep(RANDOM_INTERVAL_SECONDS)

    def calculate_distance_to_center(self, x_center_crop, y_center_crop):
        """
        타겟 중심 좌표와 캡처 영역 중심과의 거리를 계산합니다.
        """
        distance = math.dist([x_center_crop, y_center_crop], [self.CROP_CENTER_X, self.CROP_CENTER_Y])
        return distance

    def run(self):
        global CURRENT_SMOOTHING_FACTOR, ACTIVE
        time.sleep(1)
        
        gui_update_counter = 0 
        gui_update_interval = 10 
        
        overlay_update_counter = 0
        overlay_update_interval = 1 
        
        try:
            with mss.mss() as sct:
                while True:
                    loop_start_time = time.time()
                    
                    # 1. 화면 캡처 및 YOLO 추론을 위한 BGR 변환
                    sct_img = sct.grab(self.MON_CROP) 
                    frame = cv2.cvtColor(np.array(sct_img, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
                    
                    list_of_enemies = []
                    target_point = None
                    dX_err, dY_err = 0, 0 

                    # 🚨 2. YOLO 추론 실행 🚨
                    results = self.model.predict(
                        frame, 
                        save=False, 
                        classes=TARGET_CLASSES, # 설정된 타겟 클래스만 탐지
                        verbose=False, 
                        device=0 if torch.cuda.is_available() else 'cpu', 
                        half=self.half_precision,
                        imgsz=640 # 이미지 사이즈 최적화
                    )

                    positionsFrame = pd.DataFrame(
                        results[0].cpu().numpy().boxes.data, 
                        columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']
                    )
                    
                    # 3. 탐지 결과 분석 및 조준 타겟 선정
                    if not positionsFrame.empty:
                        
                        min_distance = float('inf')
                        target_row_index = -1
                        
                        for i, row in positionsFrame.iterrows():
                            # YOLO 결과는 xmin, ymin, xmax, ymax (캡처 영역 내 상대 좌표)
                            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']].astype('int')
                            
                            center_x_crop = (xmax + xmin) // 2
                            # 조준 지점(target_ratio)을 고려한 Y 좌표
                            target_y_crop = ymin + int((ymax - ymin) * self.current_target_ratio)
                            
                            distance = self.calculate_distance_to_center(center_x_crop, target_y_crop)
                            
                            if distance < min_distance:
                                min_distance = distance
                                target_row_index = i
                        
                        # 4. 오버레이 및 마우스 이동 정보 준비
                        crop_left = self.MON_CROP["left"]
                        crop_top = self.MON_CROP["top"]
                        
                        for i, row in positionsFrame.iterrows():
                            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']].astype('int')
                            
                            ww = xmax - xmin
                            hh = ymax - ymin
                            is_target = (i == target_row_index)
                            
                            abs_x = xmin + crop_left
                            abs_y = ymin + crop_top
                            
                            list_of_enemies.append((i + 1, abs_x, abs_y, ww, hh, is_target))
                            
                            if is_target:
                                target_center_x_crop = (xmax + xmin) // 2
                                target_center_y_crop = ymin + int(hh * self.current_target_ratio) 
                                
                                dX_err = target_center_x_crop - self.CROP_CENTER_X
                                dY_err = target_center_y_crop - self.CROP_CENTER_Y
                                
                                target_point = (target_center_x_crop + crop_left, target_center_y_crop + crop_top)

                    # 5. 마우스 이동 로직 (변경 없음)
                    if ACTIVE and target_point:
                        
                        if abs(dX_err) < self.DEADZONE and abs(dY_err) < self.DEADZONE:
                            dX_err = 0
                            dY_err = 0
                        
                        if dX_err != 0 or dY_err != 0:
                            
                            target_dx = dX_err * self.smoothing_factor
                            target_dy = dY_err * self.smoothing_factor
                            
                            # WMA 대신 선형 스무딩 사용
                            final_dx = target_dx # dX_err * self.smoothing_factor
                            final_dy = target_dy # dY_err * self.smoothing_factor

                            # 기존 WMA 로직 (주석 처리)
                            # final_dx = target_dx * (1 - self.smoothing_factor) + self.prev_dx * self.smoothing_factor
                            # final_dy = target_dy * (1 - self.smoothing_factor) + self.prev_dy * self.smoothing_factor

                            move_mouse_relative(final_dx, final_dy)
                            
                            self.prev_dx = final_dx
                            self.prev_dy = final_dy
                        else:
                            self.prev_dx = 0.0
                            self.prev_dy = 0.0
                    else:
                        self.prev_dx = 0.0
                        self.prev_dy = 0.0

                    # 6. UI 업데이트 (변경 없음)
                    if overlay_update_counter % overlay_update_interval == 0:
                        self.signals.update_overlay_signal.emit(list_of_enemies, target_point) 
                    overlay_update_counter += 1
                    
                    if gui_update_counter % gui_update_interval == 0:
                        self.signals.update_status_signal.emit(self.current_ratio_description, self.smoothing_factor, ACTIVE)
                        CURRENT_SMOOTHING_FACTOR = self.smoothing_factor 
                    gui_update_counter += 1
                    
                    # 7. FPS 제한 (변경 없음)
                    loop_end_time = time.time()
                    elapsed_time = loop_end_time - loop_start_time
                    sleep_time = self.target_frame_time - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as e:
            print(f"An error occurred in Detector run loop: {e}")
            self.stop_event = True


if __name__ == "__main__":
    
    with mss.mss() as sct:
        MON_FULL = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        W, H = MON_FULL["width"], MON_FULL["height"]
        
    MON_CROP = {
        "top": MON_FULL["top"] + H // 2 - CROP_HEIGHT // 2,
        "left": MON_FULL["left"] + W // 2 - CROP_WIDTH // 2,
        "width": CROP_WIDTH,
        "height": CROP_HEIGHT,
    }

    app = QtWidgets.QApplication(sys.argv) 
    
    viewer = DetectionViewer() 
    
    # 🚨 Detector 스레드 시작
    detector_thread = Detector(viewer, W, H, MON_CROP)
    viewer.set_detector_signals(detector_thread.signals)
    detector_thread.start()
    
    try:
        viewer.update_ratio_value(viewer.ratio_combo_box.currentIndex())
        viewer.update_smoothing_value(viewer.smoothing_slider.value())
        
        sys.exit(app.exec_())
    except SystemExit:
        if hasattr(detector_thread, 'overlay') and detector_thread.overlay.isVisible():
            detector_thread.overlay.close()
        if 'detector_thread' in locals() and detector_thread.is_alive():
            detector_thread.stop_event = True
        pass
