from ultralytics import YOLO
import cv2
import os
from tkinter import Tk, filedialog, Label, Button, Frame, messagebox
from PIL import Image, ImageTk, ImageOps
import pandas as pd
from collections import Counter

# YOLO 모델 로드
model = YOLO('best.pt')

# 매핑 로딩
df = pd.read_excel("C:/Users/민기/Desktop/class.cell")
brand_to_logo = dict(zip(df['Class ID'], df['Logo']))
brand_to_where = dict(zip(df['Class ID'], df['Where']))

# 전역 변수
global_img_label = None
global_logo_frame = None
logo_label = None
location_button = None
brand_label = None
most_common_name = None
most_common_id = None

# 위치 정보 출력 함수
def show_location():
    global most_common_name
    if most_common_name and most_common_name in brand_to_where:
        where = brand_to_where[most_common_name]
        messagebox.showinfo("매장 위치", f"{most_common_name} 매장은 {where}에 위치합니다.")
    else:
        messagebox.showinfo("매장 위치", "감지된 브랜드가 없거나 위치 정보가 없습니다.")

# 로고 이미지 표시 함수 (테두리 포함)
def show_detected_logo(brand_name):
    global logo_label, location_button, brand_label, most_common_id
    logo_filename = brand_to_logo.get(brand_name)
    if logo_filename:
        logo_path = os.path.join("C:/Users/민기/Desktop/capstone/logo", logo_filename)
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).resize((200, 100))
            logo_img = ImageOps.expand(logo_img, border=3, fill='gray')
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label.config(image=logo_tk)
            logo_label.image = logo_tk
            brand_label.config(text=f"감지된 브랜드: {brand_name}")
            brand_label.pack(anchor='nw')
            logo_label.pack()  # 로고 이미지 아래로 위치 이동
            location_button.pack(pady=5)  # 위치 버튼도 함께 표시
        else:
            print(f"로고 이미지 없음: {logo_path}")

# 이미지 처리 함수
def process_image_gui(image_path, output_dir, image_label, logo_frame):
    global most_common_name, most_common_id
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지 열기 실패: {image_path}")
        return

    results = model(image_path)[0]
    boxes = results.boxes

    # 0.5 이상만 필터링하여 시각화에 반영
    filtered_boxes = boxes[boxes.conf >= 0.5]
    results.boxes = filtered_boxes
    annotated = results.plot()

    # 저장
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{filename}_detected.jpg")
    cv2.imwrite(save_path, annotated)

    # PIL 이미지로 변환 및 리사이즈 후 표시
    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    max_width = 600
    max_height = 500
    w, h = pil_img.size
    ratio = min(max_width / w, max_height / h, 1.0)
    pil_img = pil_img.resize((int(w * ratio), int(h * ratio)))

    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.configure(image=tk_img)
    image_label.image = tk_img

    # 클래스별 카운트 및 로고 표시
    valid_classes = []
    for box in filtered_boxes:
        cls_id = int(box.cls[0])
        valid_classes.append(cls_id)

    if valid_classes:
        most_common_id = Counter(valid_classes).most_common(1)[0][0]
        most_common_name = model.names[most_common_id]
        show_detected_logo(most_common_name)

# 이미지 선택 함수
def select_and_process_image_gui():
    file_path = filedialog.askopenfilename(title="이미지 파일 선택", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")])
    if file_path:
        os.makedirs("output_results", exist_ok=True)
        process_image_gui(file_path, "output_results", global_img_label, global_logo_frame)

# 메인 GUI 실행
def main_gui():
    global global_img_label, global_logo_frame, logo_label, location_button, brand_label

    root = Tk()
    root.title("YOLO 로고 인식 이미지 모드")
    root.geometry("1000x600")

    btn = Button(root, text="이미지 선택", font=("맑은 고딕", 14), command=select_and_process_image_gui)
    btn.pack(pady=10)

    frame = Frame(root)
    frame.pack(fill='both', expand=True)

    global_img_label = Label(frame)
    global_img_label.pack(side='left', padx=10, pady=10)

    global_logo_frame = Frame(frame)
    global_logo_frame.pack(side='right', padx=10, pady=10)

    brand_label = Label(global_logo_frame, text="")  # 감지 전에는 표시 안 함
    logo_label = Label(global_logo_frame)
    location_button = Button(global_logo_frame, text="위치 보기", font=("맑은 고딕", 12), command=show_location)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
