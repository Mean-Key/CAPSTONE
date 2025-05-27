import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque

# ====== 설정 ======
image_path = "2400.png"
coord_txt_path = "entrance_coordinates.txt"
map_path = "map_array.npy"
scale = 0.5

# ====== 맵 불러오기 ======
map_array = np.load(map_path)

# ====== 좌표 불러오기 ======
entrances = []
with open(coord_txt_path, "r") as f:
    for line in f:
        if ":" in line:
            coord_str = line.strip().split(":")[1].strip(" ()\n")
            x, y = map(int, coord_str.split(","))
            entrances.append((int(x), int(y)))

# ====== 좌표 변환 ======
def to_map_coords(x, y, scale=0.5):
    return int(x / scale), int(y / scale)

def to_gui_coords(x, y, scale=0.5):
    return int(x * scale), int(y * scale)

# ====== BFS 경로 탐색 ======
def bfs_pathfinding(map_array, start, goal):
    h, w = map_array.shape
    visited = np.full((h, w), False, dtype=bool)
    prev = np.full((h, w, 2), -1, dtype=int)
    queue = deque([start])
    visited[start[1], start[0]] = True
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and map_array[ny, nx] == 1:
                visited[ny, nx] = True
                prev[ny, nx] = (x, y)
                queue.append((nx, ny))

    path = []
    x, y = goal
    if prev[y, x][0] == -1:
        return []  # 경로 없음
    while prev[y, x][0] != -1:
        path.append((x, y))
        x, y = prev[y, x]
    path.append(start)
    path.reverse()
    return path

# ====== 이미지 로딩 ======
img_cv = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
img_pil = Image.fromarray(img_resized)

# ====== Tkinter GUI 초기화 ======
root = tk.Tk()
root.title("앵커 기반 내 위치 → 매장 경로 GUI")

canvas = tk.Canvas(root, width=img_pil.width, height=img_pil.height)
canvas.pack()
tk_img = ImageTk.PhotoImage(img_pil)
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# ====== 입구 표시 ======
for idx, (x, y) in enumerate(entrances):
    gx, gy = to_gui_coords(x, y, scale)
    canvas.create_oval(gx - 4, gy - 4, gx + 4, gy + 4, fill="red", outline="black")
    canvas.create_text(gx, gy - 10, text=str(idx + 1), fill="gray", font=("Arial", 7, "bold"))

# ====== 입력 ======
anchor_label = tk.Label(root, text="앵커 입구 번호 3개 (예: 1,5,17):")
anchor_label.pack()
anchor_entry = tk.Entry(root)
anchor_entry.pack()

entry_label = tk.Label(root, text="도착 입구 번호:")
entry_label.pack()
entry = tk.Entry(root)
entry.pack()

def shifted_anchor(x, y, direction="down", offset=80):
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

# ====== 경로 표시 함수 ======
def show_path():
    try:
        anchor_nums = list(map(int, anchor_entry.get().split(",")))
        target = int(entry.get())

        if len(anchor_nums) != 3 or target < 1 or target > len(entrances):
            raise ValueError("입력 오류")

        # 앵커 평균 좌표 계산
        coords = [entrances[i - 1] for i in anchor_nums]
        anchors = [shifted_anchor(x, y, direction="down", offset=80) for x, y in coords]

        unique_x = list(set(x for x, _ in anchors))
        unique_y = list(set(y for _, y in anchors))

        avg_x = sum(unique_x) // len(unique_x)
        avg_y = sum(unique_y) // len(unique_y)

        anchor_gui_x = int(avg_x * scale)
        anchor_gui_y = int(avg_y * scale)

        start_map = (avg_x, avg_y)
        goal_map = entrances[target - 1]

        # 경로 탐색
        path = bfs_pathfinding(map_array, start_map, goal_map)
        if not path:
            print("경로 없음")
            return

        # 경로 시각화
        for i in range(len(path) - 1):
            x1, y1 = to_gui_coords(*path[i], scale)
            x2, y2 = to_gui_coords(*path[i + 1], scale)
            canvas.create_line(x1, y1, x2, y2, fill="cyan", width=2)

        # 타원 좌표 (GUI 기준으로 변환)
        ellipse_cx, ellipse_cy = anchor_gui_x, anchor_gui_y
        ellipse_width = 60
        ellipse_height = 30

        canvas.create_oval(ellipse_cx - ellipse_width//2, ellipse_cy - ellipse_height//2,
                           ellipse_cx + ellipse_width//2, ellipse_cy + ellipse_height//2,
                           outline="pink", fill="pink", width=1)

        canvas.create_text(ellipse_cx, ellipse_cy, text="내 위치", fill="black", font=("Arial", 8, "bold"))

    except Exception as e:
        print("오류:", e)

btn = tk.Button(root, text="경로 표시", command=show_path)
btn.pack()

root.mainloop()
