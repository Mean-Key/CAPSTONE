from ultralytics import YOLO
import cv2
import os
import pandas as pd
from collections import Counter
from tkinter import Tk, Button, Label, Frame, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

# 모델 로드
model = YOLO("best.pt")

# 브랜드 이름 및 위치 및 로고 매핑
df = pd.read_excel("C:/Users/민기/Desktop/class.cell")
brand_to_where = dict(zip(df['Class ID'], df['Where']))
brand_to_logo = dict(zip(df['Class ID'], df['Logo']))

# 전역 GUI 요소
video_label = None
brand_label = None
logo_label = None
location_button = None
most_common_name = ""
most_common_id = -1

# 위치 정보 출력 함수
def show_location():
    global most_common_name
    if most_common_name and most_common_name in brand_to_where:
        where = brand_to_where[most_common_name]
        messagebox.showinfo("매장 위치", f"{most_common_name} 매장은 {where}에 위치합니다.")
    else:
        messagebox.showinfo("매장 위치", "감지된 브랜드가 없거나 위치 정보가 없습니다.")

# 로고 이미지 표시 함수
def show_detected_logo(brand_id):
    global logo_label
    logo_filename = brand_to_logo.get(brand_id)
    if logo_filename:
        logo_path = os.path.join("C:/Users/민기/Desktop/capstone/logo", logo_filename)
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).resize((200, 100))
            logo_img = ImageOps.expand(logo_img, border=3, fill='red')
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label.config(image=logo_tk)
            logo_label.image = logo_tk
        else:
            print(f"로고 이미지 없음: {logo_path}")

# 비디오 처리 함수
def process_video(video_path):
    global most_common_name, most_common_id
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        boxes = results.boxes

        filtered_boxes = boxes[boxes.conf >= 0.5]
        results.boxes = filtered_boxes

        valid_classes = [int(box.cls[0]) for box in filtered_boxes]
        label_text = "감지된 브랜드 없음"
        if valid_classes:
            most_common_id = Counter(valid_classes).most_common(1)[0][0]
            most_common_name = model.names[most_common_id]
            label_text = f"{most_common_name} (ID: {most_common_id})"

        # 시각화 및 텍스트 삽입
        annotated_frame = results.plot()
        cv2.putText(annotated_frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 이미지 리사이즈 및 GUI 표시
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        w, h = pil_img.size
        max_width, max_height = 640, 480
        ratio = min(max_width / w, max_height / h)
        pil_img = pil_img.resize((int(w * ratio), int(h * ratio)))

        tk_img = ImageTk.PhotoImage(pil_img)
        video_label.configure(image=tk_img)
        video_label.image = tk_img
        brand_label.configure(text=label_text)

        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # 감지 종료 후 로고 및 위치 표시
    if most_common_id != -1:
        show_detected_logo(most_common_id)
        location_button.pack(pady=5)

# 비디오 선택 함수
def select_video():
    file_path = filedialog.askopenfilename(title="동영상 파일 선택", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        process_video(file_path)

# GUI 초기화
root = Tk()
root.title("YOLO 로고 인식 - 비디오 모드")
root.geometry("1100x600")

btn = Button(root, text="동영상 선택", font=("맑은 고딕", 14), command=select_video)
btn.pack(pady=10)

frame = Frame(root)
frame.pack(fill='both', expand=True)

video_label = Label(frame)
video_label.pack(side='left', padx=10, pady=10)

right_frame = Frame(frame)
right_frame.pack(side='right', padx=10, pady=10)

brand_label = Label(right_frame, text="감지된 브랜드:", font=("맑은 고딕", 13, "bold"))
brand_label.pack(anchor='n')

logo_label = Label(right_frame)
logo_label.pack(pady=5)

location_button = Button(right_frame, text="위치 보기", font=("맑은 고딕", 12), command=show_location)

root.mainloop()
