import os
from PIL import Image
import numpy as np

def generate_masks(defect_folder, mask_folder, threshold=128):
    """
    根據異常樣本生成簡單二值化遮罩。

    Args:
        defect_folder (str): 異常樣本資料夾路徑。
        mask_folder (str): 遮罩保存資料夾路徑。
        threshold (int): 二值化的閾值，0-255。
    """
    os.makedirs(mask_folder, exist_ok=True)

    for file_name in os.listdir(defect_folder):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            img_path = os.path.join(defect_folder, file_name)
            mask_path = os.path.join(mask_folder, file_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png'))
            
            # 讀取圖像並轉換為灰度
            img = Image.open(img_path).convert("L")
            
            # 二值化處理
            mask = np.array(img) > threshold
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            
            # 保存遮罩
            mask_img.save(mask_path)
            print(f"生成遮罩: {mask_path}")

# 使用範例
defect_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect_type1"
mask_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect\mask"

generate_masks(defect_folder, mask_folder)
