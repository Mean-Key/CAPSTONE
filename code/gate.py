import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# ====== 설정 ======
image_path = "2400.png"
coord_txt_path = "entrance_coordinates.txt"
scale = 0.5

# ====== 좌표 불러오기 ======
entrances = []
with open(coord_txt_path, "r") as f:
    for line in f:
        if ":" in line:
            coord_str = line.strip().split(":")[1].strip(" ()\n")
            x, y = map(int, coord_str.split(","))
            entrances.append((int(x * scale), int(y * scale)))

# ====== 이미지 로딩 및 축소 ======
img_cv = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
img_pil = Image.fromarray(img_resized)

# ====== Tkinter 초기화 ======
root = tk.Tk()
root.title("입구 앵커 기반 평균 좌표 표시 GUI")

canvas = tk.Canvas(root, width=img_pil.width, height=img_pil.height)
canvas.pack()

tk_img = ImageTk.PhotoImage(img_pil)
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# ====== 점 + 텍스트 표시 ======
for idx, (x, y) in enumerate(entrances):
    r = 4
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="black")
    canvas.create_text(x, y - 10, text=str(idx + 1), fill="grey", font=("Arial", 7, "bold"))

# ====== 사용자 입력 필드 ======
entry_label = tk.Label(root, text="입구 번호 3개 (예: 1,5,17):")
entry_label.pack()

entry = tk.Entry(root)
entry.pack()

# ====== 입구에서 앵커 좌표 계산 ======
def shifted_anchor(x, y, direction="down", offset=40):
    if direction == "down":
        return x, y + offset
    elif direction == "up":
        return x, y - offset
    elif direction == "left":
        return x - offset, y
    elif direction == "right":
        return x + offset, y
    else:
        return x, y

# ====== 앵커 기반 평균 좌표 표시 함수 ======
def show_anchor_average():
    try:
        nums = list(map(int, entry.get().split(",")))
        if len(nums) != 3:
            raise ValueError("3개 번호를 입력하세요.")
        coords = [entrances[i - 1] for i in nums]

        # 방향은 전부 "down"으로 가정 (매장별 설정은 추후 추가 가능)
        anchors = [shifted_anchor(x, y, direction="down", offset=40) for x, y in coords]

        unique_x = list(set(x for x, _ in anchors))
        unique_y = list(set(y for _, y in anchors))

        avg_x = sum(unique_x) // len(unique_x)
        avg_y = sum(unique_y) // len(unique_y)

        # 타원 중심 좌표
        cx, cy = avg_x, avg_y

        # 타원 너비와 높이 설정 (픽셀 단위)
        ellipse_width = 60
        ellipse_height = 30

        canvas.create_oval(cx - ellipse_width//2, cy - ellipse_height//2,
                           cx + ellipse_width//2, cy + ellipse_height//2,
                           outline="pink", fill="pink", width=1)

        canvas.create_text(cx, cy, text="내 위치", fill="black", font=("Arial", 8, "bold"))

    except Exception as e:
        print("입력 오류:", e)

btn = tk.Button(root, text="앵커 중앙 표시", command=show_anchor_average)
btn.pack()

root.mainloop()
