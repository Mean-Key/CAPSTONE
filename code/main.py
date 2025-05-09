import tkinter as tk
from tkinter import messagebox
import subprocess

# 실행할 파일 이름
MODE_SCRIPTS = {
    "이미지 인식": "YOLO_LOGO_img.py",
    "비디오 인식": "YOLO_LOGO_video.py",
    "실시간 인식": "YOLO_LOGO_Webcam.py"
}

# Tkinter UI
window = tk.Tk()
window.title("YOLO 로고 인식 실행기")
window.geometry("400x250")

label = tk.Label(window, text="실행할 모드를 선택하세요", font=("맑은 고딕", 14))
label.pack(pady=20)

# 선택 변수
mode_var = tk.StringVar()
mode_var.set("웹캠 실시간 감지")  # 기본 선택

# 라디오 버튼으로 모드 선택
for mode in MODE_SCRIPTS.keys():
    tk.Radiobutton(window, text=mode, variable=mode_var, value=mode, font=("맑은 고딕", 12)).pack(anchor="w", padx=40)

# 실행 함수
def run_selected_script():
    mode = mode_var.get()
    script = MODE_SCRIPTS[mode]
    try:
        subprocess.run(["python", script], check=True)
    except Exception as e:
        messagebox.showerror("실행 오류", f"{script} 실행 중 오류 발생:\n{e}")

# 실행 버튼
run_btn = tk.Button(window, text="선택한 모드 실행", command=run_selected_script, font=("맑은 고딕", 12))
run_btn.pack(pady=30)

window.mainloop()
