
-----

## 🎮 Overwatch AI 에임 보조 프로그램 사용법 요약

### 1\. ⚙️ 준비물 (Pre-requisites)

이 코드를 실행하려면 다음과 같은 환경과 파일이 준비되어 있어야 합니다.

  * **NVIDIA GPU:** TensorRT와 CUDA를 사용하므로 NVIDIA 그래픽 카드가 필수입니다.
  * **필수 라이브러리 설치:**
    ```bash
    pip install opencv-python numpy mss keyboard pynvml torch ultralytics pywin32
    ```
  * **모델 파일:**
      * **`overwatch.pt`:** 학습된 YOLO 모델 파일 (필수).
      * **`overwatch_tensorrt.engine`:** TensorRT 엔진 파일 (선택적이지만 권장).

### 2\. 🔑 주요 조작 키

| 키 (Key) | 기능 (Function) | 설명 (Description) |
| :--- | :--- | :--- |
| **`F2`** | **AI On/Off 토글** | 에임 보조 기능 전체를 켜거나 끕니다. |
| **`F3`** | **모드 토글** | 마우스 이동 모드(Sense / Human)를 전환합니다. |
| **`Q`** | **프로그램 종료** | 디버그 창이 활성화된 상태에서 `Q`를 누르거나, 디버그 창을 닫으면 종료됩니다. |

-----

## 💻 코드의 상세 작동 방식

### 1\. 🧠 모델 및 환경 설정

  * **`convert_to_tensorrt()` 함수:**
      * `overwatch.pt` YOLO 모델 파일을 \*\*TensorRT 엔진 파일(`overwatch_tensorrt.engine`)\*\*로 변환합니다.
      * TensorRT는 NVIDIA GPU에서 딥러닝 모델의 추론(Inference) 속도를 극대화하여 **매우 빠른 실시간 감지**를 가능하게 합니다.
      * 엔진 파일이 이미 있으면 변환 과정을 건너뜁니다.
  * **캡처 영역:**
      * 화면 전체가 아닌, 중앙을 기준으로 **350x350 픽셀** 영역 (`MONITOR_WIDTH`, `MONITOR_HEIGHT`)만 캡처합니다. (오버워치 십자선 주변만 빠르게 스캔)
  * **모델 로딩:**
      * TensorRT 엔진 파일이 있으면 이를 사용하고, 없으면 일반 `.pt` 파일을 사용합니다.

### 2\. 🖼️ 화면 캡처 및 디버그

  * **`capture_screen()` 함수 (MSS 사용):**
      * 화면 중앙의 정의된 영역(`MONITOR_LEFT`, `MONITOR_TOP`)을 빠르게 캡처하여 NumPy 배열(`img`)로 반환합니다.
  * **디버그 창 (`DEBUG_WINDOW_NAME`):**
      * 캡처된 화면을 \*\*`Overwatch AI`\*\*라는 이름의 창에 실시간으로 표시하며, 항상 \*\*최상위(Always On Top)\*\*로 설정됩니다.
      * 이 창에는 AI의 상태(ACTIVE/INACTIVE), 모드(Human/Sense), 그리고 감지된 적의 바운딩 박스와 조준점(빨간색 원)이 표시됩니다.

### 3\. 🎯 조준 및 마우스 이동 (`aim_at_target` & `move_mouse`)

  * **타겟 필터링:**
      * `TARGET_CLASS` (클래스 0, 아마도 "적 플레이어")인 객체만 선택합니다.
      * `CONF_THRESHOLD` (0.5) 이상의 신뢰도(Confidence)를 가진 객체만 유효한 타겟으로 간주합니다.
  * **최적의 타겟 선택:**
      * 감지된 타겟들 중 **화면 중앙(십자선)으로부터 가장 가까운** 타겟을 선택합니다. (`np.argmin(distances)`)
      * `MAX_DISTANCE` (450) 이내에 있는 타겟만 유효합니다.
  * **조준점 보정 (`target_center_y`):**
      * 일반적인 박스 중앙이 아니라, **박스의 위쪽 1/6 지점**을 조준점으로 잡습니다. 이는 오버워치 캐릭터의 **'머리' 또는 '목' 부분**을 조준하기 위한 오프셋(Head/Neck Offset)으로 보입니다.
      * `target_center_y = (y1 + (y2 - y1) / 6).astype(int)`
  * **마우스 이동 (`move_mouse`):**
      * 현재 마우스 위치와 목표 조준점(`global_target_x, global_target_y`) 사이의 \*\*차이(`delta_x`, `delta_y`)\*\*를 계산합니다.
      * 이동량에 \*\*`ACCUMULATED_X/Y_ERROR`\*\*를 더해 미세한 소수점 이동 오차를 누적 보정합니다.
  * **모드별 이동:**
      * **Human Mode (활성화):** `SMOOTH_FACTOR = 25`로 설정되어 **매우 부드럽고 느리게** 마우스를 움직여 사람처럼 보이게 합니다.
      * **Sense Mode (비활성화):** `sensitivity = 0.3`으로 설정되어 **더 빠르고 정확하게** 마우스를 움직입니다.
-----
교육용입니다 적대로 사용하지 마세요.
