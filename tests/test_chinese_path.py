# test_chinese_path.py
# 测试中文路径是否正常工作

import cv2
import numpy as np
from PIL import Image
import os

def test_cv2_imread_chinese_path():
    """测试 OpenCV 读取中文路径"""
    print("=" * 60)
    print("测试 OpenCV 读取中文路径")
    print("=" * 60)
    
    # 测试路径
    test_path = r"D:\植物面积还原算法\src\1\1.jpg"
    
    print(f"\n测试文件: {test_path}")
    print(f"文件是否存在: {os.path.exists(test_path)}")
    
    # 方法 1: 直接使用 cv2.imread (可能不支持中文)
    print("\n方法 1: cv2.imread()")
    img1 = cv2.imread(test_path)
    if img1 is not None:
        print(f"✅ 成功读取，图像尺寸: {img1.shape}")
    else:
        print("❌ 读取失败 (中文路径不支持)")
    
    # 方法 2: 使用 np.fromfile + cv2.imdecode (支持中文)
    print("\n方法 2: np.fromfile + cv2.imdecode()")
    try:
        img_data = np.fromfile(test_path, dtype=np.uint8)
        img2 = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img2 is not None:
            print(f"✅ 成功读取，图像尺寸: {img2.shape}")
        else:
            print("❌ 读取失败")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
    
    # 方法 3: PIL Image.open (支持中文)
    print("\n方法 3: PIL Image.open()")
    try:
        img3 = Image.open(test_path)
        print(f"✅ 成功读取，图像尺寸: {img3.size}")
        # 转换为 OpenCV 格式
        img3_cv = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2BGR)
        print(f"   转换为 OpenCV 格式: {img3_cv.shape}")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("- 如果方法 1 失败，说明 OpenCV 不支持中文路径")
    print("- 方法 2 和 3 都支持中文路径")
    print("- 推荐使用方法 2 (已在代码中应用)")
    print("=" * 60)


def test_cv2_imwrite_chinese_path():
    """测试 OpenCV 保存到中文路径"""
    print("\n" + "=" * 60)
    print("测试 OpenCV 保存到中文路径")
    print("=" * 60)
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试路径（中文）
    output_path = r"D:\植物面积还原算法\测试图像.jpg"
    
    # 方法 1: 直接使用 cv2.imwrite (可能不支持中文)
    print(f"\n方法 1: cv2.imwrite()")
    print(f"保存路径: {output_path}")
    success1 = cv2.imwrite(output_path, test_img)
    if success1 and os.path.exists(output_path):
        print(f"✅ 保存成功")
        os.remove(output_path)  # 清理
    else:
        print(f"❌ 保存失败 (中文路径不支持)")
    
    # 方法 2: 使用 cv2.imencode + tofile (支持中文)
    print(f"\n方法 2: cv2.imencode + tofile()")
    try:
        is_success, buffer = cv2.imencode('.jpg', test_img)
        if is_success:
            buffer.tofile(output_path)
            if os.path.exists(output_path):
                print(f"✅ 保存成功")
                os.remove(output_path)  # 清理
            else:
                print(f"❌ 文件未创建")
        else:
            print(f"❌ 编码失败")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("- 如果方法 1 失败，说明 OpenCV 不支持保存到中文路径")
    print("- 方法 2 支持中文路径")
    print("- 推荐使用方法 2 (已在代码中应用)")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "🔧 中文路径兼容性测试 🔧\n")
    
    test_cv2_imread_chinese_path()
    test_cv2_imwrite_chinese_path()
    
    print("\n✅ 所有测试完成！")
    print("\n💡 提示:")
    print("   如果发现问题，项目中的所有代码已更新为支持中文路径的方法")
    print("   - camera_calibration.py: 已修复")
    print("   - pnp.py: 已修复")
    print("   - Image_segmentation_prediction.py: PIL 自带支持\n")
