# Image_segmentation_prediction.py
# DeeplabV3 语义分割模型封装，用于植物叶子区域检测

import os
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import cv2


class DeeplabV3:
    """
    DeeplabV3 语义分割模型封装类
    用于植物叶子区域的精确分割和面积计算
    """
    
    def __init__(self, model_path: str, mix_type: int = 1):
        """
        初始化 DeeplabV3 模型
        
        参数:
        - model_path: str, 模型权重文件路径 (.pth 文件)
        - mix_type: int, 模型配置类型 (默认为 1)
        
        注意:
        - 自动检测并使用 GPU (如果可用)
        - 模型输入为 RGB 图像，输出为二值分割掩码
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载预训练的 DeeplabV3 模型 (使用 ResNet-101 作为骨干网络)
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=2)
        
        # 加载自定义训练的权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # 处理可能的 state_dict 格式 (某些保存方式可能包含 'model' 键)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict, strict=False)
            print(f"成功加载模型权重: {model_path}")
        except Exception as e:
            print(f"警告: 加载模型权重失败 ({e})，使用随机初始化权重")
        
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        self.mix_type = mix_type
        
        # 图像预处理参数 (ImageNet 标准化)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        """
        预处理输入图像
        
        参数:
        - pil_img: PIL.Image, 输入图像 (RGB 格式)
        
        返回:
        - tensor: torch.Tensor, 预处理后的张量 (1, 3, H, W)
        
        处理步骤:
        1. 调整图像大小 (保持长宽比，最大边为 512)
        2. 归一化到 [0, 1]
        3. 标准化 (使用 ImageNet 均值和标准差)
        4. 转换为 PyTorch 张量
        """
        # 转换为 NumPy 数组
        img = np.array(pil_img).astype(np.float32) / 255.0
        
        # 调整大小 (可选，用于加速推理)
        # h, w = img.shape[:2]
        # max_size = 512
        # if max(h, w) > max_size:
        #     scale = max_size / max(h, w)
        #     img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # 标准化
        img = (img - self.mean) / self.std
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 转为 tensor 并添加 batch 维度
        tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        
        return tensor
    
    def detect_image(self, pil_img: Image.Image):
        """
        对输入图像进行语义分割，检测植物叶子区域
        
        参数:
        - pil_img: PIL.Image, 输入图像 (RGB 格式)
        
        返回:
        - mask_img: PIL.Image, 分割掩码图像 (二值图，叶子区域为白色)
        - leaf_area: int, 叶子像素面积
        
        算法原理:
        1. 预处理输入图像
        2. 使用 DeeplabV3 模型进行语义分割
        3. 提取叶子类别 (假设类别 1 为叶子)
        4. 后处理: 形态学操作去除噪声
        5. 计算叶子像素面积
        
        误差来源:
        - 模型分割精度 (训练数据质量影响)
        - 光照、遮挡导致的误分类
        - 图像模糊降低边缘检测准确性
        """
        original_size = pil_img.size  # (width, height)
        
        # 预处理
        input_tensor = self.preprocess_image(pil_img)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)['out']  # (1, num_classes, H, W)
            # 获取每个像素的预测类别
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)
        
        # 调整掩码大小为原始图像大小
        pred = cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建二值掩码 (假设类别 1 为叶子，0 为背景)
        mask = (pred == 1).astype(np.uint8) * 255
        
        # 形态学后处理: 去除小噪点，填充小孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填孔
        
        # 计算叶子面积 (像素数)
        leaf_area = int(np.sum(mask > 0))
        
        # 转换为 PIL 图像
        mask_img = Image.fromarray(mask, mode='L')  # 灰度图像
        
        return mask_img, leaf_area


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试 DeeplabV3 模型的分割功能
    
    使用方法:
    python Image_segmentation_prediction.py
    """
    import sys
    
    # 模型路径
    model_path = r"D:\fixcrmdif\area_restore_test0814\models\plant_image_recognition20250717.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请将模型文件放置在正确路径，或修改 model_path 变量")
        sys.exit(1)
    
    # 初始化模型
    deeplab = DeeplabV3(model_path, mix_type=1)
    
    # 测试图像路径
    test_image_path = r"D:\植物面积还原算法\src\1\1.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像不存在: {test_image_path}")
        print("请提供有效的测试图像路径")
        sys.exit(1)
    
    # 加载测试图像
    img = Image.open(test_image_path)
    print(f"测试图像尺寸: {img.size}")
    
    # 执行分割
    mask, area = deeplab.detect_image(img)
    print(f"检测到的叶子面积: {area} 像素")
    
    # 保存分割结果
    mask.save("test_mask.png")
    print("分割掩码已保存到: test_mask.png")
    
    # 可视化结果
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"分割掩码 (面积: {area} 像素)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_result.png")
    print("可视化结果已保存到: test_result.png")
    plt.show()
