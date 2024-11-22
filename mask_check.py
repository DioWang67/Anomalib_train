# import os

# def check_defect_mask_alignment(defect_folder, mask_folder):
#     """
#     檢查異常樣本與遮罩對應關係。

#     Args:
#         defect_folder (str): 異常樣本資料夾路徑。
#         mask_folder (str): 遮罩資料夾路徑。
#     """
#     defect_files = sorted([f for f in os.listdir(defect_folder) if f.endswith('.png')])
#     mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('_mask.png')])

#     print("異常樣本數:", len(defect_files))
#     print("遮罩文件數:", len(mask_files))

#     unmatched = []
#     for defect_file in defect_files:
#         mask_file = defect_file.replace('.png', '_mask.png')
#         if mask_file not in mask_files:
#             unmatched.append(defect_file)
    
#     if unmatched:
#         print("以下異常樣本沒有對應的遮罩文件:")
#         print("\n".join(unmatched))
#     else:
#         print("所有異常樣本與遮罩文件對應正確。")

# # 修改為你的實際路徑
# defect_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect"
# mask_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect\mask"

# check_defect_mask_alignment(defect_folder, mask_folder)

from PIL import Image
import numpy as np
import os

def check_mask_content(mask_folder):
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # 檢查是否只有0和255
        unique_values = np.unique(mask)
        if not set(unique_values).issubset({0, 255}):
            print(f"遮罩文件 {mask_file} 包含非0或255的像素值: {unique_values}")
        else:
            print(f"遮罩文件 {mask_file} 格式正確。")

# 修改為你的遮罩資料夾路徑
mask_folder = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect\mask"
check_mask_content(mask_folder)
