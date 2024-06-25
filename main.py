import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput

from ultralytics import YOLO
import pygetwindow as gw
import pynput
import time

# 初始化YOLOv8模型
model = YOLO("best.pt")

# 锁头功能开关
trace_on = False
detection_modes = {
    "teamCT": {"classes": [2, 3]},
    "teamT": {"classes": [0, 1]},
    "Solo": {"classes": [0, 1, 2, 3]}
}
team_mode = "Solo"
team_cur = detection_modes[team_mode]
auto_shoot = False

resize = (1080, 720)
scale_ratio_x = 1
scale_ratio_y = 1


# 截取屏幕
def capture_window():
    window_title = "Counter-Strike 2"
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"无法找到名为'{window_title}'的游戏窗口")
    window = windows[0]
    return window


def on_press(key):
    global team_cur, team_mode, auto_shoot, trace_on
    try:
        if key == pynput.keyboard.Key.tab:
            trace_on = not trace_on
            print(f"自动移动功能：{trace_on}")
        elif key == pynput.keyboard.Key.f5:
            team_mode = "teamCT"
            team_cur = detection_modes[team_mode]
            print(f"当前阵营: {team_mode}")
        elif key == pynput.keyboard.Key.f6:
            team_mode = "teamT"
            team_cur = detection_modes[team_mode]
            print(f"当前阵营: {team_mode}")
        elif key == pynput.keyboard.Key.f7:
            team_mode = "Solo"
            team_cur = detection_modes[team_mode]
            print(f"当前阵营: {team_mode}")
    except AttributeError:
        pass


def mouse_move(boxes):
    global scale_ratio_x, scale_ratio_y
    distance_list = []
    window = capture_window()
    if window is None:
        return

    current_x, current_y = pyautogui.position()

    start_time = time.time()
    for box in boxes:
        box_xyxy = box.xyxy.tolist()[0]
        x1 = int(box_xyxy[0] * scale_ratio_x)
        x2 = int(box_xyxy[2] * scale_ratio_x)
        y1 = int(box_xyxy[1] * scale_ratio_y)
        y2 = int(box_xyxy[3] * scale_ratio_y)

        cls = int(box.cls.tolist()[0])
        target_box_position = [
            window.left + x1,
            window.top + y1,
            window.left + x2,
            window.top + y2
        ]
        target_box_centre = (window.left + (x1 + x2) // 2, window.top + (y1 + y2) // 2)
        eu_distance = (current_x - target_box_centre[0]) ** 2 + (current_y - target_box_centre[1]) ** 2
        distance_list.append([cls, target_box_position, target_box_centre, eu_distance])

    if distance_list:
        target_x = 0
        target_y = 0
        distance_list.sort(key=lambda x: x[3])
        if distance_list[0][0] == 1 or distance_list[0][0] == 3:       # cls == 1: ct_head; cls == 3: t_head
            target_x = int(distance_list[0][2][0])
            target_y = int(distance_list[0][2][1])
        else:
            if len(distance_list) > 1:
                if distance_list[1][0] == 1 or distance_list[1][0] == 3:
                    target_x = int(distance_list[1][2][0])
                    target_y = int(distance_list[1][2][1])
            else:
                target_x = int(distance_list[0][2][0])
                target_y = int(distance_list[0][1][1])

        if not target_x and not target_y:
            return

        if trace_on:
            distance_x = target_x - current_x
            distance_y = target_y - current_y
            delta_x = int(distance_x)
            delta_y = int(distance_y)
            # pyautogui.moveRel(delta_x, delta_y, duration=0.15, tween=pyautogui.easeInOutQuad)
            pydirectinput.moveRel(delta_x + 10, delta_y, relative=True)

    end_time = time.time()

    time_leap = end_time - start_time
    print(f"耗时{time_leap}")


def detect():
    global scale_ratio_x, scale_ratio_y, resize
    window = capture_window()
    if window is None:
        return
    with mss.mss() as sct:
        monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        scale_ratio_x = window.width / resize[0]
        scale_ratio_y = window.height / resize[1]
        img_resize = cv2.resize(img, resize)
        results = model(img_resize, classes=team_cur["classes"], device=0, conf=0.6)
        result = results[0]

        show_window = result.plot()
        cv2.imshow("YOLOv8 on CS2", show_window)

        if result.boxes:
            mouse_move(result.boxes)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            exit(0)


if __name__ == "__main__":
    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()
    while True:
        detect()

    cv2.destroyAllWindows()
    listener.stop()

