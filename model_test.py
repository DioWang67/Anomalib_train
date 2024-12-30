import torch
from pathlib import Path
import numpy as np
import cv2
from anomalib.models.image.fastflow.torch_model import FastflowModel
from torchvision import transforms
import traceback

def test_single_image(model_path, image_path):
    """
    使用訓練好的 Fastflow 模型對單張圖片進行異常檢測

    Args:
        model_path: 模型權重文件路徑
        image_path: 待測試圖片路徑
    """
    # 設置設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 初始化模型，使用與訓練時相同的參數
    model = FastflowModel(
        backbone="resnet18",
        pre_trained=True,
        flow_steps=8,
        conv3x3_only=False,
        hidden_ratio=1.0,
        input_size=(256, 256)
    ).to(device)

    # 加載檢查點
    checkpoint = torch.load(model_path, map_location=device)

    # 獲取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 調整鍵名稱以匹配模型的 state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k.replace('model.', '')
        else:
            new_key = k
        new_state_dict[new_key] = v

    # 加載權重
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 讀取並預處理圖片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"無法讀取圖片: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # 使用與訓練時相同的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    print(f"輸入張量形狀: {input_tensor.shape}")

    # 進行預測
    with torch.no_grad():
        outputs = model(input_tensor)

    # 獲取異常圖和分數
    anomaly_map = outputs.squeeze().cpu().numpy()

    # 計算異常分數（例如，使用最大值）
    anomaly_score = float(np.max(anomaly_map))
    print(f"預測結果形狀: {anomaly_map.shape}")

    # 將異常圖調整為原始圖片大小
    anomaly_map_resized = cv2.resize(
        anomaly_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    # 方法1：只對高異常值部分進行正規化
    percentile = 40 # 調整此值以控制顯著區域
    threshold_value = np.percentile(anomaly_map_resized, percentile)
    anomaly_map_filtered = np.clip(anomaly_map_resized, threshold_value, anomaly_map_resized.max())

    normalized_anomaly_map = cv2.normalize(
        anomaly_map_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # 方法2：應用閾值過濾低異常值（可選，根據需要選擇使用）
    # threshold_value = 0.5 * anomaly_map_resized.max()
    # anomaly_map_thresholded = np.where(anomaly_map_resized >= threshold_value, anomaly_map_resized, 0)
    # normalized_anomaly_map = cv2.normalize(
    #     anomaly_map_thresholded, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    # )

    # 選擇更突出差異的顏色映射
    heatmap = cv2.applyColorMap(normalized_anomaly_map, cv2.COLORMAP_HOT)

    # 調整疊加權重，使異常區域更突出
    overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)

    # 保存結果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "anomaly_map.png"), normalized_anomaly_map)
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)

    # 根據閾值判斷是否為異常
    threshold = -0.3  # 根據實際情況調整
    is_anomaly = anomaly_score > threshold

    return {
        "label": is_anomaly,
        "score": anomaly_score,
        "anomaly_map": anomaly_map_resized,
        "visualization": overlay
    }

if __name__ == "__main__":
    model_path = r"D:\Git\robotlearning\Anomalib_train\model.ckpt"
    image_path = r"D:\Git\robotlearning\Anomalib_train\cropped\163631_con1_0.jpg"
# D:\Git\robotlearning\Anomalib_train\datasets\con1\train\good\131435_con1_0_aug_16.jpg
# D:\Git\robotlearning\Anomalib_train\cropped\163631_con1_0.jpg
    try:
        print("開始執行預測...")
        results = test_single_image(model_path, image_path)

        # 輸出結果
        print(f"\n預測標籤: {'異常' if results['label'] else '正常'}")
        print(f"異常分數: {results['score']:.4f}")
        print("\n已保存結果圖片於 'output' 資料夾：")
        print("- anomaly_map.png")
        print("- overlay.png")
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")
        traceback.print_exc()
