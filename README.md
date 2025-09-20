
### **✨ Py-Aimbot 소개 및 사용법 (Introduction & How to Use Py-Aimbot)**

이 스크립트는 게임에서 특정 색상(밝은 빨간색)의 적을 자동으로 조준하는 파이썬 프로그램이에요. 사용법은 아주 간단해요.

This script is a Python program that automatically aims at targets of a specific color (bright red) in games. It's very simple to use.

1.  **실행하기 (Run the Script)**: 스크립트를 실행하면 터미널(검은색 창)이 열리고 "your monitors (ex: 1920 1080):"라는 메시지가 나와요.

    When you run the script, a terminal (a black window) will open with the message "your monitors (ex: 1920 1080):".

    
3.  **해상도 입력 (Enter Resolution)**: 현재 사용 중인 모니터의 해상도(예: `1920 1080`)를 입력하고 엔터를 누르면 돼요.

    Simply enter your monitor's resolution (e.g., `1920 1080`) and press Enter.

    

5.  **작동 시작 (Start Operation)**: 에임봇이 바로 작동을 시작하고, 화면 중앙에 있는 빨간색 대상을 찾아서 마우스를 움직여줘요.
   
    The aimbot will start working immediately, finding red targets in the center of your screen and moving the mouse for you.

    

7.  **종료하기 (Quit)**: 프로그램을 멈추고 싶을 때는 터미널 창을 클릭하고 키보드 **`q`**를 누르면 돼요.

    To stop the program, click on the terminal window and press the **`q`** key.

    

---

### **⚠️ 사용 시 주의할 점 (Important Warnings)**

이 스크립트는 편리하지만, 몇 가지 조심해야 할 점이 있어요.

While this script is useful, there are a few important things to be careful about.


1.  **게임 밴(Ban) 위험 (Risk of Game Ban)**: 많은 온라인 게임은 외부 프로그램(에임봇, 매크로 등) 사용을 금지하고 있어요. 이 스크립트를 온라인 게임에서 사용하면 **계정이 영구 정지될 수 있어요.** 이 스크립트는 오프라인 환경이나 테스트용으로만 사용하는 것이 가장 안전해요.

    Many online games prohibit the use of external programs (aimbots, macros, etc.). Using this script in an online game can result in a **permanent account ban.** It's safest to only use this script in an offline environment or for testing purposes.

    

4.  **정확도 문제 (Accuracy Issues)**: 에임봇은 설정된 색상을 인식해서 움직이기 때문에, 다른 빨간색 물체(맵, UI 요소 등)가 화면에 보이면 **잘못 조준할 수 있어요.** 게임 화면의 디자인에 따라 정확도가 크게 달라질 수 있어요.

    The aimbot recognizes and moves based on a set color, so it may **aim incorrectly** if other red objects (e.g., on the map or in the UI) appear on the screen. The accuracy can vary significantly depending on the game's interface design.

    

7.  **시스템 성능 (System Performance)**: 이 스크립트는 실시간으로 화면을 캡처하고 분석하기 때문에, 컴퓨터의 CPU나 GPU 성능에 따라 **게임이 느려지거나 렉(Lag)이 발생할 수 있어요.** 최적의 성능을 위해 `BOX_SIZE` 값을 적절하게 조절하는 것이 좋아요.

    This script captures and analyzes the screen in real-time, which might cause **your game to slow down or lag** depending on your computer's CPU or GPU performance. It's a good idea to adjust the `BOX_SIZE` value to optimize performance.

    

---

### **🧐 스크립트 상세 설명 (Detailed Script Explanation)**

코드를 보면 다양한 기능들이 서로 유기적으로 연결되어 있어요.

Looking at the code, you'll see various functions are interconnected.

* **`main()` 함수 (The `main()` function)**: 이 부분이 모든 것을 움직이는 엔진이에요. `while True:` 루프를 사용해서 에임봇을 끊임없이 동작시키죠.

    This is the engine that drives everything. It uses a `while True:` loop to keep the aimbot running continuously.

  

* **`grab_screen()` 함수 (The `grab_screen()` function)**: **`mss`**라는 라이브러리를 사용해서 모니터 화면을 엄청 빠르게 찍어내는 역할을 해요. 특히, 화면 전체가 아닌 `BOX_SIZE` 만큼의 작은 영역만 캡처해서 성능을 높여요.

    Using a library called **`mss`**, this function captures the monitor screen very quickly. It improves performance by capturing only a small area defined by `BOX_SIZE` instead of the entire screen.

  

* **`process_image()` 함수 (The `process_image()` function)**: 캡처된 이미지에서 목표물을 찾는 역할을 해요. `cv2.inRange()`를 사용해 빨간색을 찾아내고, `cv2.findContours()`로 윤곽선을 찾아서 `move_mouse()` 함수를 호출해요.
  
    This function finds the target in the captured image. It uses `cv2.inRange()` to find red areas, `cv2.findContours()` to locate contours, and then calls the `move_mouse()` function.

  

* **`move_mouse()` 함수 (The `move_mouse()` function)**: 마우스를 실제로 움직이는 부분이에요. **`human_mode`**가 `True`일 때는 부드럽게 움직이고, `False`일 때는 목표물로 바로 이동해요.
  
    This is the part that actually moves the mouse. When **`human_mode`** is `True`, it moves smoothly, and when it's `False`, it moves directly to the target.

  
만약 코드를 직접 수정해보고 싶다면, `BOX_SIZE` 값을 변경해서 인식 범위를 조절하거나, `LOWER_COLOR_RANGE`와 `UPPER_COLOR_RANGE` 값을 변경해서 다른 색상을 추적하도록 바꿀 수도 있어요. 

If you want to modify the code yourself, you can change the `BOX_SIZE` value to adjust the detection area or alter the `LOWER_COLOR_RANGE` and `UPPER_COLOR_RANGE` values to track a different color.

---
