import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import pandas as pd
import os

# 엑셀에서 매핑 불러오기
mapping_file = "C:/Users/민기/Desktop/class.cell"
df = pd.read_excel(mapping_file)
brand_to_logo = dict(zip(df['Class ID'], df['Logo']))
brand_to_where = dict(zip(df['Class ID'], df['Where']))

# 모델 및 상태 변수
model = YOLO("best.pt")

running = False
paused = False
cap = None
out = None
class_counter = defaultdict(int)
class_buffer = deque(maxlen=30)
last_detected_classes = []
most_common_name = None

# Tkinter UI 구성
window = tk.Tk()
window.title("YOLO 실시간 감지")
window.geometry("1050x700")

video_label = Label(window)
video_label.grid(row=0, column=0, rowspan=20, padx=10, pady=10)

result_label = Label(window, text="감지된 클래스:\n", justify="left", font=("Arial", 14))
result_label.grid(row=0, column=1, sticky="nw", padx=10)

logo_label = Label(window)
logo_label.grid(row=1, column=1, sticky="nw", padx=10, pady=10)

# 버튼 프레임
button_frame = tk.Frame(window)
button_frame.grid(row=10, column=1, pady=20, sticky="n")

Button(button_frame, text="감지 시작", command=lambda: start_detection()).grid(row=0, column=0, padx=5)
Button(button_frame, text="감지 정지", command=lambda: stop_detection()).grid(row=0, column=1, padx=5)
Button(button_frame, text="프레임 정지", command=lambda: pause_frame()).grid(row=0, column=2, padx=5)
Button(button_frame, text="위치 출력", command=lambda: show_location()).grid(row=1, column=0, columnspan=3, pady=10)

# 로고 이미지 표시 함수 (테두리 포함)
def show_detected_logo(brand_name):
    logo_filename = brand_to_logo.get(brand_name)
    if logo_filename:
        logo_path = os.path.join("C:/Users/민기/Desktop/capstone/logo", logo_filename)
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).resize((200, 100))
            logo_img = ImageOps.expand(logo_img, border=5, fill='red')  # 빨간 테두리
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label.config(image=logo_tk)
            logo_label.image = logo_tk
        else:
            print(f"로고 이미지 없음: {logo_path}")

# 위치 정보 출력 함수
def show_location():
    if most_common_name and most_common_name in brand_to_where:
        where = brand_to_where[most_common_name]
        messagebox.showinfo("매장 위치", f"{most_common_name} 매장은 {where}에 위치합니다.")
    else:
        messagebox.showinfo("매장 위치", "감지된 브랜드가 없거나 위치 정보가 없습니다.")

# 프레임 갱신 함수
def update_frame():
    global cap, running, paused, class_counter, out

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
                    shown.append(f"{model.names[cls_id]} → {count}회 감지")
            result_text = "현재 감지 요약:\n" + ("\n".join(shown) if shown else " 감지 없음")
            result_label.config(text=result_text)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            if out:
                out.write(annotated_frame)

        video_label.after(30, update_frame)

# 컨트롤 함수들
def start_detection():
    global cap, running, paused, out
    if not running:
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (width, height)) 
        running = True
        paused = False
        update_frame()

def stop_detection():
    global cap, running, paused, out
    running = False
    paused = False
    if cap:
        cap.release()
        cap = None
    if out:
        out.release() 
        out = None
    video_label.config(image='')
    result_label.config(text="감지된 클래스:\n")
    logo_label.config(image='')

def pause_frame():
    global running, paused, last_detected_classes, most_common_name
    if running:
        paused = True
        running = False
        last_detected_classes = [
            model.names[cls_id]
            for cls_id, count in class_counter.items()
            if count >= 3
        ]
        print("감지 정지됨. 감지된 클래스:", last_detected_classes)
        if class_counter:
            most_common_cls_id = max(class_counter.items(), key=lambda x: x[1])[0]
            most_common_name = model.names[most_common_cls_id]
            show_detected_logo(most_common_name)

# 종료 처리
def on_close():
    stop_detection()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)
window.mainloop()
