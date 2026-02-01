from ultralytics import YOLO
import cv2

print("=" * 60)
print("YOLO 模型测试")
print("=" * 60)

model = YOLO(r'D:\植物面积还原算法\models\plant.pt')  # 加载模型
print("✅ 模型加载成功")

test_image = r'D:\植物面积还原算法\src\1\1.jpg'
print(f"\n测试图像: {test_image}")

results = model(test_image, verbose=True)  # 进行预测，显示详细信息

print("\n" + "=" * 60)
print("检测结果:")
print("=" * 60)

for i, result in enumerate(results):
    boxes = result.boxes  # 获取检测到的边界框
    
    if boxes is not None and len(boxes) > 0:
        print(f"\n✅ 检测到 {len(boxes)} 个目标")
        print(f"\n目标详情:")
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0]) if box.cls is not None else -1
            
            print(f"  目标 {j+1}:")
            print(f"    边界框: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            print(f"    置信度: {conf:.3f}")
            print(f"    类别ID: {cls}")
            print(f"    宽度×高度: {x2-x1:.1f} × {y2-y1:.1f}")
    else:
        print("\n❌ 未检测到任何目标！")
        print("\n可能的原因:")
        print("  1. 模型未训练识别该类型的植物")
        print("  2. 图像质量不佳（模糊、光照不足）")
        print("  3. 置信度阈值过高")
        print("  4. 模型路径或图像路径错误")
    
    # 显示带标注的图像
    print(f"\n正在显示结果图像...")
    result.show()  # 显示结果
    
    # 同时保存结果
    output_path = r'D:\植物面积还原算法\yolo_result.jpg'
    print(f"✅ 结果已保存到: {output_path}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

