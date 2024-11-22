import os
from PIL import Image

root_dir = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\train\good"

for file in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file)
    try:
        with Image.open(file_path) as img:
            img.verify()  # 驗證圖片
        print(f"Image {file} is valid.")
    except Exception as e:
        print(f"Error with file {file}: {e}")
