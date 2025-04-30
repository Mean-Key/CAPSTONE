import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import pandas as pd
import os


# 엑셀에서 매핑 불러오기
mapping_file = "C:/Users/민기/Desktop/class.cell"  # 엑셀 경로 수정
df = pd.read_excel(mapping_file)
brand_to_logo = dict(zip(df['Class ID'], df['Logo']))  # 클래스명 → 로고파일명


# 모델 및 상태 변수
model = YOLO("best.pt")

running = False
paused = False
cap = None
class_counter = defaultdict(int)
class_buffer = deque(maxlen=30)
last_detected_classes = []


# Tkinter UI 구성
window = tk.Tk()
window.title("YOLO 실시간 감지")
window.geometry("1050x650")

video_label = Label(window)
video_label.place(x=10, y=10)

result_label = Label(window, text="감지된 클래스:\n", justify="left", font=("Arial", 14))
result_label.place(x=720, y=20)

logo_label = Label(window)  # 로고 이미지 출력용
logo_label.place(x=720, y=200)


# 로고 이미지 표시 함수
def show_detected_logo(brand_name):
    logo_filename = brand_to_logo.get(brand_name)
    if logo_filename:
        logo_path = os.path.join("C:/Users/민기/Desktop/capstone/logo", logo_filename)
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).resize((200, 100))
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label.config(image=logo_tk)
            logo_label.image = logo_tk
        else:
            print(f"로고 이미지 없음: {logo_path}")


# 프레임 갱신 함수
def update_frame():
    global cap, running, paused, class_counter

    if running and not paused:
        ret, frame = cap.read()
        if ret:
            results = model(frame)
            annotated_frame = results[0].plot()

            current_classes = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf >= 0.5:
                    current_classes.append(cls_id)

            class_buffer.append(current_classes)
            class_counter.clear()
            for frame_classes in class_buffer:
                for cls_id in frame_classes:
                    class_counter[cls_id] += 1

            shown = []
            for cls_id, count in class_counter.items():
                if count >= 3:
                    shown.append(f"{model.names[cls_id]} → {count}번 감지됨")
            result_text = "현재 감지 요약:\n" + ("\n".join(shown) if shown else "감지 없음")
            result_label.config(text=result_text)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        video_label.after(30, update_frame)


# 컨트롤 함수들
def start_detection():
    global cap, running, paused
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        paused = False
        update_frame()

def stop_detection():
    global cap, running, paused
    running = False
    paused = False
    if cap:
        cap.release()
        video_label.config(image='')
        result_label.config(text="감지된 클래스:\n")
        logo_label.config(image='')

def pause_frame():
    global running, paused, last_detected_classes
    if running:
        paused = True
        running = False
        last_detected_classes = [
            model.names[cls_id]
            for cls_id, count in class_counter.items()
            if count >= 3
        ]
        print("🛑 감지 정지됨. 감지된 클래스:", last_detected_classes)
        if class_counter:
            most_common_cls_id = max(class_counter.items(), key=lambda x: x[1])[0]
            most_common_name = model.names[most_common_cls_id]
            show_detected_logo(most_common_name)


# 버튼
Button(window, text="감지 시작", command=start_detection).place(x=720, y=400)
Button(window, text="감지 정지", command=stop_detection).place(x=820, y=400)
Button(window, text="프레임 정지", command=pause_frame).place(x=920, y=400)


# 종료 처리
def on_close():
    stop_detection()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)
window.mainloop()
