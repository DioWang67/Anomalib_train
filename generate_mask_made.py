import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import os
from glob import glob

class MaskGenerator:
    def __init__(self):
        self.drawing = False
        self.points = []  # 當前多邊形的點
        self.polygons = []  # 所有已完成的多邊形
        self.current_image_index = 0
        self.image_files = []
        
        # 創建主視窗
        self.root = tk.Tk()
        self.root.title("Select Anomalies")
        
        # 創建頂部框架用於顯示當前圖片信息
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.image_label = tk.Label(self.info_frame, text="Current image: ")
        self.image_label.pack(side=tk.LEFT)
        
        # 創建畫布
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 創建底部框架用於按鈕
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 綁定滑鼠事件
        self.canvas.bind("<Button-1>", self.add_point)  # 左鍵添加點
        self.canvas.bind("<Button-3>", self.complete_polygon)  # 右鍵完成多邊形
        self.canvas.bind("<Motion>", self.mouse_move)  # 滑鼠移動更新預覽線
        
        # 創建按鈕
        save_button = tk.Button(self.button_frame, text="Save (S)", command=self.save_mask)
        save_button.pack(side=tk.LEFT, padx=5)
        
        next_button = tk.Button(self.button_frame, text="Next (N)", command=self.next_image)
        next_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(self.button_frame, text="Clear (C)", command=self.clear_canvas)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        undo_button = tk.Button(self.button_frame, text="Undo (U)", command=self.undo_last)
        undo_button.pack(side=tk.LEFT, padx=5)
        
        quit_button = tk.Button(self.button_frame, text="Quit (Q)", command=self.quit)
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # 綁定鍵盤事件
        self.root.bind('s', lambda e: self.save_mask())
        self.root.bind('n', lambda e: self.next_image())
        self.root.bind('c', lambda e: self.clear_canvas())
        self.root.bind('u', lambda e: self.undo_last())
        self.root.bind('q', lambda e: self.quit())
        
        # 用於存儲臨時線條的ID
        self.temp_line_id = None
        
    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        
        # 畫點
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="red")
        
        # 如果有兩個以上的點，畫線連接
        if len(self.points) > 1:
            x1, y1 = self.points[-2]
            x2, y2 = self.points[-1]
            self.canvas.create_line(x1, y1, x2, y2, fill="green", width=2)
            
    def mouse_move(self, event):
        # 如果有至少一個點，顯示預覽線
        if self.points:
            if self.temp_line_id:
                self.canvas.delete(self.temp_line_id)
            x1, y1 = self.points[-1]
            x2, y2 = event.x, event.y
            self.temp_line_id = self.canvas.create_line(x1, y1, x2, y2, fill="gray", dash=(4, 4))
            
    def complete_polygon(self, event):
        if len(self.points) > 2:
            # 連接最後一個點和第一個點
            x1, y1 = self.points[-1]
            x2, y2 = self.points[0]
            self.canvas.create_line(x1, y1, x2, y2, fill="green", width=2)
            
            # 保存多邊形並清除當前點
            self.polygons.append(self.points.copy())
            self.points = []
            
            # 清除臨時預覽線
            if self.temp_line_id:
                self.canvas.delete(self.temp_line_id)
                
    def clear_canvas(self):
        # 清除畫布上的所有內容
        self.canvas.delete("all")
        self.points = []
        self.polygons = []
        self.load_current_image()  # 重新載入當前圖片
        
    def undo_last(self):
        if self.points:
            # 如果正在畫多邊形，移除最後一個點
            self.points.pop()
            self.canvas.delete("all")
            self.load_current_image()
            # 重新繪製當前的多邊形
            self.redraw_current_state()
        elif self.polygons:
            # 如果沒有正在畫的點，移除最後一個完成的多邊形
            self.polygons.pop()
            self.canvas.delete("all")
            self.load_current_image()
            # 重新繪製所有多邊形
            self.redraw_current_state()
            
    def redraw_current_state(self):
        # 重新繪製所有已完成的多邊形
        for poly in self.polygons:
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                self.canvas.create_line(x1, y1, x2, y2, fill="green", width=2)
                self.canvas.create_oval(x1-2, y1-2, x1+2, y1+2, fill="red")
                
        # 重新繪製當前正在畫的多邊形
        for i, (x, y) in enumerate(self.points):
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="red")
            if i > 0:
                x1, y1 = self.points[i-1]
                self.canvas.create_line(x1, y1, x, y, fill="green", width=2)
    
    def save_mask(self):
        if hasattr(self, 'current_image_path'):
            # 生成掩碼圖檔名
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            mask_name = f"{base_name}_mask.png"
            output_path = os.path.join(self.output_dir, mask_name)
            
            # 創建空白掩碼圖
            mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
            
            # 轉換座標比例
            scale_x = self.image_shape[1] / self.canvas.winfo_width()
            scale_y = self.image_shape[0] / self.canvas.winfo_height()
            
            # 填充所有多邊形區域
            for polygon in self.polygons:
                # 轉換座標
                scaled_polygon = np.array([
                    [int(x * scale_x), int(y * scale_y)] 
                    for x, y in polygon
                ], np.int32)
                
                # 填充多邊形
                cv2.fillPoly(mask, [scaled_polygon], 255)
            
            cv2.imwrite(output_path, mask)
            messagebox.showinfo("Success", f"Mask saved as: {mask_name}")
            
            # 自動進入下一張圖片
            self.next_image()
    
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.points = []
            self.polygons = []
        else:
            messagebox.showinfo("完成", "已處理完所有圖片！")
            self.quit()
    
    def load_current_image(self):
        self.current_image_path = self.image_files[self.current_image_index]
        self.image_label.config(text=f"Current image: {os.path.basename(self.current_image_path)}")
        
        # 清除之前的內容
        self.canvas.delete("all")
        
        # 讀取並顯示新圖片
        image = cv2.imread(self.current_image_path)
        self.image_shape = image.shape
        
        # 轉換為 PIL 圖片
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 調整圖片大小以適應螢幕
        screen_width = self.root.winfo_screenwidth() - 100
        screen_height = self.root.winfo_screenheight() - 100
        
        # 計算縮放比例
        ratio = min(screen_width / image.shape[1], screen_height / image.shape[0])
        new_width = int(image.shape[1] * ratio)
        new_height = int(image.shape[0] * ratio)
        
        # 調整畫布大小
        self.canvas.config(width=new_width, height=new_height)
        
        # 調整圖片大小並顯示
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
    def quit(self):
        self.root.quit()
        
    def generate_masks(self, input_dir, output_dir):
        # 獲取輸入目錄中的所有圖片
        self.image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            self.image_files.extend(glob(os.path.join(input_dir, ext)))
        self.image_files.sort()
        
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the input directory!")
            return
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 載入第一張圖片
        self.current_image_index = 0
        self.load_current_image()
        
        print("使用說明：")
        print("- 左鍵點擊: 添加多邊形頂點")
        print("- 右鍵點擊: 完成當前多邊形")
        print("- 's' 鍵: 保存當前掩碼圖")
        print("- 'n' 鍵: 切換到下一張圖片")
        print("- 'c' 鍵: 清除當前所有框選")
        print("- 'u' 鍵: 撤銷上一步操作")
        print("- 'q' 鍵: 退出程式")
        
        self.root.mainloop()

def generate_masks(input_dir, output_dir):
    app = MaskGenerator()
    app.generate_masks(input_dir, output_dir)

# 範例用法
if __name__ == "__main__":
    input_dir = r"S:\DioWang\robotlearning\img\con_for_anomalib\ground_truth"
    output_dir = r"S:\DioWang\robotlearning\img\con_for_anomalib\con_ground_truth_mask"
    generate_masks(input_dir, output_dir)