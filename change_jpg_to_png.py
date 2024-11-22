import os
from PIL import Image

def convert_jpg_to_png(input_folder, output_folder):
    """
    將指定資料夾內的 .jpg 圖片轉換為 .png 並存儲到輸出資料夾。

    Args:
        input_folder (str): 輸入資料夾路徑。
        output_folder (str): 輸出資料夾路徑。
    """
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍歷輸入資料夾的所有檔案
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(".jpg"):
            jpg_path = os.path.join(input_folder, file_name)
            png_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".png")
            
            try:
                # 開啟 .jpg 圖片並轉換為 .png 格式
                with Image.open(jpg_path) as img:
                    img.save(png_path, "PNG")
                print(f"已成功將 {jpg_path} 轉換為 {png_path}")
            except Exception as e:
                print(f"無法處理 {jpg_path}：{e}")

# 使用範例
input_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect_type1"  # 替換為你的輸入資料夾路徑
output_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect_type1"  # 替換為你的輸出資料夾路徑

convert_jpg_to_png(input_folder, output_folder)
