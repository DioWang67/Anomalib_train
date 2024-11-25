import torch
from pathlib import Path
import numpy as np
import cv2
from anomalib.models.image.fastflow.torch_model import FastflowModel
from torchvision import transforms

def test_single_image(model_path, image_path):
    """
    使用训练好的 Fastflow 模型对单张图片进行异常检测

    Args:
        model_path: 模型权重文件路径
        image_path: 待测试图片路径
    """
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化模型，使用与训练时相同的参数
    model = FastflowModel(
        backbone="resnet18",
        pre_trained=True,
        flow_steps=8,
        conv3x3_only=False,
        hidden_ratio=1.0,
        input_size=(256,256)
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 调整键名称以匹配模型的 state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k.replace('model.', '')
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # 加载权重
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # 读取并预处理图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # 使用与训练时相同的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 进行预测
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # 获取异常图和分数
    anomaly_map = outputs.squeeze().cpu().numpy()
    
    # 计算异常分数（例如，使用最大值）
    anomaly_score = float(np.max(anomaly_map))
    
    print(f"预测结果形状: {anomaly_map.shape}")
    
    # 将异常图调整为原始图片大小
    anomaly_map_resized = cv2.resize(
        anomaly_map,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    # 规范化异常图以用于可视化
    normalized_anomaly_map = cv2.normalize(
        anomaly_map_resized, 
        None, 
        0, 
        255, 
        cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )
    
    # 创建热图
    heatmap = cv2.applyColorMap(normalized_anomaly_map, cv2.COLORMAP_JET)
    
    # 叠加热图到原始图片
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # 保存结果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "anomaly_map.png"), normalized_anomaly_map)
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)
    
    # 根据阈值判断是否为异常
    threshold = 0.5  # 根据实际情况调整
    is_anomaly = anomaly_score > threshold
    
    return {
        "label": is_anomaly,
        "score": anomaly_score,
        "anomaly_map": anomaly_map_resized,
        "visualization": overlay
    }

if __name__ == "__main__":
    model_path = r"D:\Git\robotlearning\Anomalib_train\results\Fastflow\MVTec\con1\v14\weights\lightning\model.ckpt"
    image_path = r"D:\Git\robotlearning\Anomalib_train\datasets\con1\test\defect_type_1\165245_con1_0.png"
    
    try:
        print("开始执行预测...")
        results = test_single_image(model_path, image_path)
        
        # 输出结果
        print(f"\n预测标签: {'异常' if results['label'] else '正常'}")
        print(f"异常分数: {results['score']:.4f}")
        print("\n已保存结果图片于 'output' 文件夹：")
        print("- anomaly_map.png")
        print("- overlay.png")
        
    except Exception as e:
        print(f"执行时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
