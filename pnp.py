# auto_scale_report_yolo_9x6_nosign.py
# 功能：基于 YOLO 和 DeeplabV3 检测植物，结合 9x6 棋盘格标定，计算角度/距离/高度变化下的还原面积，生成对比表
# 特点：强制使用名为 '1' 的图片作为标定参考，输出旋转向量 (rvec) 和平移向量 (tvec)


import os  # 用于文件和目录操作，如列出图片文件
import cv2  # OpenCV 库，处理图像、棋盘格检测和姿态估计
import numpy as np  # 数值计算库，处理矩阵和数组操作
import pandas as pd  # 数据处理库，生成和保存 CSV 表格
from PIL import Image  # 处理图像输入（PIL 格式）
from Image_segmentation_prediction import DeeplabV3  # DeeplabV3 模型，用于植物分割
from ultralytics import YOLO  # YOLO 模型，用于植物检测

# ========== 模型路径 ==========
# 定义 YOLO 和 DeeplabV3 模型文件的路径
YOLO_MODEL_PATH = r"D:\个人项目\植物面积还原算法\models\plant.pt"  # YOLO 模型权重路径
DEEPLAB_MODEL_PATH = r"D:\个人项目\植物面积还原算法\models\plant_image_recognition20250821.pth"  # DeeplabV3 模型权重路径


# ========== YOLO + Deeplab 封装 ==========
class YoloDeeplabWrapper:
    """封装 YOLO 和 DeeplabV3 模型，先用 YOLO 检测植物边界框，再用 DeeplabV3 分割计算像素面积"""

    def __init__(self, yolo_path: str, deeplab_path: str):
        """
        初始化 YOLO 和 DeeplabV3 模型

        参数:
        - yolo_path: str, YOLO 模型路径
        - deeplab_path: str, DeeplabV3 模型路径

        返回:
        无

        注意:
        - 加载预训练模型，用于植物检测和分割
        - 错误来源: 模型路径错误或文件损坏可能导致加载失败
        """
        self.yolo = YOLO(yolo_path)  # 加载 YOLO 模型
        self.deeplab = DeeplabV3(deeplab_path, mix_type=1)  # 加载 DeeplabV3 模型，mix_type=1 表示特定配置

    def detect_and_segment(self, pil_img, filename=None):
        """
        对输入图像进行 YOLO 检测和 DeeplabV3 分割，计算植物像素面积

        参数:
        - pil_img: PIL.Image, 输入图像（RGB 格式）
        - filename: str (可选), 文件名，用于调试保存掩码

        返回:
        - annotated_img: PIL.Image, 带绿色掩码和边界框的标注图像
        - areas: list[dict], 每个检测物体的像素面积列表，形如 [{'area': int}, ...]

        算法原理:
        - YOLO 检测植物边界框，DeeplabV3 分割叶子区域，二值化后计算像素面积
        - 数学公式: leaf_area = np.sum(mask_np > 0)
        - 误差来源: YOLO 边界框偏差（光照、遮挡），DeeplabV3 分割误差 5-10%（模糊图像）
        - 注意: 保存调试掩码到 debug_mask_*.png，标注图像添加 0.4 透明度绿色掩码
        """
        img_np = np.array(pil_img)[:, :, ::-1]  # 将 PIL 图像（RGB）转为 NumPy 数组（BGR）供 OpenCV 处理
        results = self.yolo(img_np, verbose=False)[0]  # 使用 YOLO 检测，关闭日志输出以提高效率

        # 调试信息：显示检测到的目标数量
        num_detections = len(results.boxes) if results.boxes is not None else 0
        if filename:
            print(f"  [调试] {filename}: YOLO 检测到 {num_detections} 个目标")
            if num_detections > 0:
                print(f"  [调试] 置信度: {[float(box.conf) for box in results.boxes]}")

        areas = []  # 存储每个检测物体的像素面积
        annotated = img_np.copy()  # 复制图像，用于添加标注（边界框和掩码）
        box_index = 0  # 边界框索引，用于调试文件名

        for box in results.boxes:  # 遍历 YOLO 检测到的每个边界框
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 获取边界框坐标并转为整数
            crop_img = pil_img.crop((x1, y1, x2, y2))  # 裁剪边界框区域，供 DeeplabV3 分割
            r_image, area_pixels = self.deeplab.detect_image(crop_img)  # 使用 DeeplabV3 分割裁剪区域

            # 处理分割掩码，确保只显示叶子面积
            mask_np = np.array(r_image)  # 将分割结果（PIL 图像）转为 NumPy 数组
            if mask_np.ndim == 3:  # 如果掩码是 RGB，转换为单通道灰度
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            # 二值化掩码，叶子区域为 255，非叶子区域为 0
            _, mask_np = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

            # 调整掩码大小以匹配裁剪区域
            mask_np = cv2.resize(mask_np, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            # 计算叶子像素面积
            leaf_area = int(np.sum(mask_np > 0))  # 统计掩码中非零像素数作为面积
            areas.append({'area': leaf_area})  # 保存面积到列表

            # 保存调试掩码（如果提供了文件名）
            if filename:
                cv2.imwrite(f"debug_mask_{filename}_box{box_index}.png", mask_np)  # 保存掩码图像用于调试
            box_index += 1  # 递增边界框索引

            # 创建绿色半透明掩码，仅覆盖叶子像素
            # 检查是否有有效的掩码像素
            if leaf_area > 0:
                overlay = np.zeros_like(annotated[y1:y2, x1:x2])  # 初始化与裁剪区域相同的空白覆盖层
                overlay[mask_np > 0] = [0, 255, 0]  # 在叶子像素处填充绿色
                alpha = 0.4  # 设置掩码透明度
                region = annotated[y1:y2, x1:x2].copy()  # 复制裁剪区域以避免修改原图
                # 混合原图和绿色掩码，透明度为 alpha
                region[mask_np > 0] = cv2.addWeighted(
                    region[mask_np > 0], 1 - alpha,
                    overlay[mask_np > 0], alpha, 0.0
                )
                annotated[y1:y2, x1:x2] = region  # 将混合结果放回原图

            # 绘制 YOLO 边界框（绿色，线宽 2 像素）
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return Image.fromarray(annotated[:, :, ::-1]), areas  # 返回标注图像（BGR 转 RGB）和面积列表


# ========== 棋盘格参数 ==========
CHECKERBOARD = (9, 6)  # 定义棋盘格尺寸：9 列 x 6 行
square_size = 1  # 定义棋盘格单格边长（cm）

# ========== 加载相机内参 ==========
# 从 NPZ 文件加载相机标定结果
calib_data = np.load('camera_params.npz')  # 加载标定文件
camera_matrix = calib_data['camera_matrix']  # 获取内参矩阵 (3x3)，包含焦距和主点
dist_coeffs = calib_data['dist_coeffs']  # 获取畸变系数 (1x5)，用于校正镜头畸变

# ========== 初始化模型 ==========
model = YoloDeeplabWrapper(YOLO_MODEL_PATH, DEEPLAB_MODEL_PATH)  # 实例化 YOLO+DeeplabV3 模型

# ========== 创建世界坐标 ==========
# 初始化棋盘格角点的世界坐标 (x, y, z)，z=0 表示平面
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # 生成 9x6 网格坐标
objp *= square_size  # 按边长缩放坐标，单位 cm


# ========== 工具函数 ==========
def get_scale_pnp(img_path: str) -> (float, float, np.ndarray, np.ndarray):
    """
    通过棋盘格计算缩放因子（scale）、俯仰角（pitch）、旋转向量（rvec）和平移向量（tvec）

    参数:
    - img_path: str, 图像路径

    返回:
    - scale: float, 平均像素边长
    - pitch_deg: float, 俯仰角（度）
    - rvec: np.ndarray, 旋转向量 (3x1)
    - tvec: np.ndarray, 平移向量 (3x1)

    算法原理:
    - 使用 cv2.solvePnP 估计摄像头姿态，cv2.projectPoints 投影世界坐标到图像平面，计算平均像素边长
    - 数学公式: projected = camera_matrix * [R | t] * objp; scale = mean(dist_i), dist_i = sqrt((u_i - u_{i+1})^2 + (v_i - v_{i+1})^2)
    - 误差来源: 内参偏差 (RMSE=0.635),  透视失真 (高度 >15cm, 角度 >8°), 角点检测误差 (光照模糊)
    - 注意: 若未检测到棋盘格，抛出 ValueError
    """
    # 读取图像（支持中文路径）
    try:
        img_data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")  # 检查图像是否成功加载
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像，便于棋盘格检测
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 检测棋盘格角点
    if not ret:
        raise ValueError(f"未检测到棋盘格：{img_path}")  # 检查是否找到角点
    # 精化角点位置，提高检测精度
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # 使用 PnP 算法估计摄像头姿态
    _, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    # 投影棋盘格世界坐标到图像平面
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)  # 重塑投影点为 (N, 2) 数组
    rows, cols = CHECKERBOARD[1], CHECKERBOARD[0]  # 获取棋盘格行数和列数
    dists = []  # 存储相邻投影点间的像素距离
    for i in range(rows):  # 遍历棋盘格行
        for j in range(cols):  # 遍历棋盘格列
            idx = i * cols + j  # 计算当前角点索引
            if j + 1 < cols:  # 计算水平相邻点距离
                dists.append(np.linalg.norm(projected[idx] - projected[idx + 1]))
            if i + 1 < rows:  # 计算垂直相邻点距离
                dists.append(np.linalg.norm(projected[idx] - projected[idx + cols]))
    scale = np.mean(dists)  # 计算平均像素边长

    # 从旋转向量计算俯仰角（pitch）
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转为旋转矩阵
    pitch = np.arcsin(-R[2, 0])  # 计算俯仰角（弧度），公式：pitch = arcsin(-R[2,0])

    return scale, np.degrees(pitch), rvec, tvec  # 返回缩放因子、俯仰角（度）、旋转向量和平移向量


def get_scale_pixel(img_path: str) -> float:
    """
    通过棋盘格角点计算像素间距缩放因子（scale）

    参数:
    - img_path: str, 图像路径

    返回:
    - scale: float, 平均像素边长

    算法原理:
    - 检测棋盘格角点，计算相邻角点的平均像素距离
    - 数学公式: scale = mean(dist_i), dist_i = sqrt((u_i - u_{i+1})^2 + (v_i - v_{i+1})^2)
    - 误差来源: 角点检测偏差 (光照模糊), 透视失真 (高度变化时格子变形), 数值精度 (小像素距离时不准)
    - 注意: 若未检测到棋盘格，抛出 ValueError
    """
    # 读取图像（支持中文路径）
    try:
        img_data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")  # 检查图像是否成功加载
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 检测棋盘格角点
    if not ret:
        raise ValueError(f"未检测到棋盘格：{img_path}")  # 检查是否找到角点
    # 精化角点位置，提高检测精度
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    dists = []  # 存储相邻角点间的像素距离
    for i in range(len(corners2) - 1):  # 遍历角点
        p1, p2 = corners2[i][0], corners2[i + 1][0]  # 获取相邻角点坐标
        dists.append(np.linalg.norm(p1 - p2))  # 计算欧几里得距离
    return np.mean(dists)  # 返回平均像素边长


# ========== 主处理函数 ==========
def process_folder(folder: str):
    """
    处理文件夹中的图像，计算参考数据、面积和变化量，生成对比结果

    参数:
    - folder: str, 图片文件夹路径

    返回:
    - results: dict, 按维度 ('delta_angle', 'delta_distance', 'delta_height') 存储 DataFrame
    - ref_max_area: float, 参考图片的最大植物像素面积

    算法原理:
    - 检测参考图片 (`1.jpg`)，计算所有图片的面积、姿态和缩放，生成三种方法的还原面积和误差
    - 数学公式: 还原面积 = actual_area * (scale_length)^2; 误差 = (还原面积 / ref_max_area - 1) * 100
    - 误差来源:  姿态估计误差 (rvec, tvec 不准), 分割精度 (YOLO+DeeplabV3 5-10%)
    - 注意: 强制使用 `1.jpg` 作为标定参考，若未找到抛出 ValueError；显示标注图像需手动按键继续
    """
    # 列出文件夹中所有 jpg/png/jpeg 格式的图像文件
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 寻找标定图片（名为 '1'，忽略扩展名）
    ref_file = None
    for f in files:
        base = os.path.splitext(f)[0]  # 获取文件名（无扩展名）
        if base == '1':
            ref_file = f  # 找到标定图片
            break
    if ref_file is None:
        raise ValueError("未找到名为'1'的标定图片（例如'1.jpg'）。")  # 检查是否找到标定图片

    ref_path = os.path.join(folder, ref_file)  # 构造标定图片完整路径
    # 计算标定图片的缩放因子、俯仰角和姿态
    ref_sp, ref_pitch_deg, ref_rvec, ref_tvec = get_scale_pnp(ref_path)
    ref_pitch_rad = np.radians(ref_pitch_deg)  # 将俯仰角从度转为弧度
    ref_sx = get_scale_pixel(ref_path)  # 计算标定图片的像素间距缩放因子

    # 处理标定图片以获取参考面积
    ref_img = Image.open(ref_path)  # 打开标定图片（PIL 格式）
    _, ref_areas = model.detect_and_segment(ref_img, filename=ref_file)  # 使用 YOLO+DeeplabV3 检测和分割
    if not ref_areas:
        raise ValueError(f"标定图片 {ref_file} 没有检测到植物")  # 检查是否检测到植物
    ref_max_area = max(ref_areas, key=lambda x: x['area'])['area']  # 获取最大植物像素面积

    records = []  # 存储每张图片的处理结果
    for f in files:  # 遍历所有图片
        path = os.path.join(folder, f)  # 构造图片路径
        try:
            img = Image.open(path)  # 打开图片（PIL 格式）
        except:
            print(f"跳过无法打开的图片: {f}")  # 跳过无法加载的图片
            continue
        # 使用 YOLO+DeeplabV3 检测和分割，获取标注图像和面积
        annotated_img, areas = model.detect_and_segment(img, filename=f)

        # 显示分割结果，调整分辨率为 1280x720
        annotated_np = np.array(annotated_img)[:, :, ::-1]  # 将 PIL 图像（RGB）转为 BGR
        annotated_np = cv2.resize(annotated_np, (1280, 720))  # 调整图像大小以便显示
        cv2.imshow(f"Segmentation Result - {f}", annotated_np)  # 显示标注图像（带边界框和掩码）
        cv2.waitKey(0)  # 等待用户按键继续
        cv2.destroyAllWindows()  # 关闭显示窗口

        if not areas:  # 检查是否检测到植物
            print(f"{f} 没有检测到植物，跳过")
            continue

        max_area = max(areas, key=lambda x: x['area'])['area']  # 获取最大植物像素面积

        try:
            # 计算当前图片的缩放因子、俯仰角和姿态
            sp, pitch_deg, rvec, tvec = get_scale_pnp(path)
            sx = get_scale_pixel(path)  # 计算像素间距缩放因子
        except ValueError as e:
            print(f"棋盘格检测失败，跳过: {f} -> {e}")  # 跳过棋盘格检测失败的图片
            continue

        pitch_rad = np.radians(pitch_deg)  # 将俯仰角从度转为弧度
        delta_t = tvec - ref_tvec  # 计算平移向量差
        delta_height = delta_t[2, 0]  # 获取高度变化（z 轴，单位 cm）
        delta_distance = np.linalg.norm(delta_t[:2, 0])  # 获取水平距离变化（x-y 平面，单位 cm）
        delta_angle = pitch_deg - ref_pitch_deg  # 获取角度变化（度）

        # 计算缩放因子（基于 PnP 和像素间距）
        length_scale_pnp = sp / square_size if square_size != 0 else 1  # PnP 缩放因子
        length_scale_pixel = sx / square_size if square_size != 0 else 1  # 像素间距缩放因子

        # 将 rvec 和 tvec 转为字符串以便存储
        rvec_str = ','.join([f"{x:.4f}" for x in rvec.flatten()])
        tvec_str = ','.join([f"{x:.4f}" for x in tvec.flatten()])
        ref_rvec_str = ','.join([f"{x:.4f}" for x in ref_rvec.flatten()])
        ref_tvec_str = ','.join([f"{x:.4f}" for x in ref_tvec.flatten()])

        # 记录图片处理结果
        records.append(dict(
            filename=f,  # 图片文件名
            delta_height=delta_height,  # 高度变化
            delta_distance=delta_distance,  # 距离变化
            delta_angle=delta_angle,  # 角度变化
            actual_area=max_area,  # 实际像素面积
            length_scale_pnp=length_scale_pnp,  # PnP 缩放因子
            length_scale_pixel=length_scale_pixel,  # 像素间距缩放因子
            pitch_rad=pitch_rad,  # 俯仰角（弧度）
            rvec=rvec_str,  # 旋转向量（字符串）
            tvec=tvec_str,  # 平移向量（字符串）
            ref_rvec=ref_rvec_str,  # 标定图片旋转向量
            ref_tvec=ref_tvec_str  # 标定图片平移向量
        ))

    df = pd.DataFrame(records)  # 将记录转为 DataFrame

    results = {}  # 存储按维度排序的结果
    for key in ['delta_angle', 'delta_distance', 'delta_height']:  # 遍历三种变化维度
        sub = df.sort_values(by=key).reset_index(drop=True)  # 按指定维度排序
        if sub.empty:
            continue  # 跳过空数据
        # 计算校正缩放因子
        sub[f'{key}_scale_length_pnp'] = ref_sp / sub['length_scale_pnp']  # PnP 缩放因子
        sub[f'{key}_scale_length_pixel'] = ref_sx / sub['length_scale_pixel']  # 像素间距缩放因子
        # 添加姿态+角度补偿缩放因子
        sub[f'{key}_scale_length_pnp_angle'] = sub[f'{key}_scale_length_pnp'] * (
                np.cos(ref_pitch_rad) / np.cos(sub['pitch_rad']))
        # 计算三种方法的还原面积
        sub['还原面积_pnp'] = sub['actual_area'] * (sub[f'{key}_scale_length_pnp'] ** 2)
        sub['还原面积_pixel'] = sub['actual_area'] * (sub[f'{key}_scale_length_pixel'] ** 2)
        sub['还原面积_pnp_angle'] = sub['actual_area'] * (sub[f'{key}_scale_length_pnp_angle'] ** 2)
        # 计算三种方法的误差
        sub['差异_pnp'] = ((sub['还原面积_pnp'] / ref_max_area) - 1) * 100
        sub['差异_pixel'] = ((sub['还原面积_pixel'] / ref_max_area) - 1) * 100
        sub['差异_pnp_angle'] = ((sub['还原面积_pnp_angle'] / ref_max_area) - 1) * 100
        results[key] = sub  # 保存结果

    return results, ref_max_area  # 返回结果字典和参考面积


# ========== 命令行入口 ==========
if __name__ == "__main__":
    import argparse  # 用于解析命令行参数

    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--folder", required=True, help="图片文件夹路径")  # 添加文件夹路径参数
    parser.add_argument("--mode", choices=['h', 'd', 'r'], required=True,
                        help="选择一个变化维度: h（高度）, d（距离）, r（角度）")  # 添加变化维度参数
    args = parser.parse_args()  # 解析命令行参数

    dim_map = {'h': 'delta_height', 'd': 'delta_distance', 'r': 'delta_angle'}  # 映射命令行参数到维度
    chosen_key = dim_map[args.mode]  # 获取选定的变化维度

    results, ref_max_area = process_folder(args.folder)  # 处理文件夹中的图片

    if chosen_key not in results:
        print(f"没有检测到 {chosen_key} 维度的数据")  # 检查是否有对应维度数据
        exit()  # 退出程序

    df = results[chosen_key].reset_index(drop=True)  # 获取选定维度的结果并重置索引
    df = df.loc[:, ~df.columns.duplicated()]  # 移除重复列

    # 创建转置表格，包含所有关键数据
    transpose_table = {
        "序号": list(range(1, len(df) + 1)),  # 生成序号列
        "文件名": df['filename'].tolist(),  # 文件名列表
        "高度变化": [f"{h:.2f}cm" for h in df['delta_height']],  # 高度变化（格式化为 cm）
        "距离变化": [f"{d:.2f}cm" for d in df['delta_distance']],  # 距离变化（格式化为 cm）
        "角度变化": [f"{a:.2f}°" for a in df['delta_angle']],  # 角度变化（格式化为度）
        "旋转向量": df['rvec'].tolist(),  # 旋转向量（字符串）
        "平移向量": df['tvec'].tolist(),  # 平移向量（字符串）
        "标定旋转向量": df['ref_rvec'].tolist(),  # 标定图片旋转向量
        "标定平移向量": df['ref_tvec'].tolist(),  # 标定图片平移向量
        "姿态变化缩放因子": df[f'{chosen_key}_scale_length_pnp'].round(4).tolist(),  # PnP 缩放因子（保留 4 位小数）
        "像素间距缩放因子": df[f'{chosen_key}_scale_length_pixel'].round(4).tolist(),  # 像素间距缩放因子
        "姿态变化+角度补偿缩放因子": df[f'{chosen_key}_scale_length_pnp_angle'].round(4).tolist(),  # 姿态+角度补偿缩放因子
        "实际面积": df['actual_area'].round(0).astype(int).tolist(),  # 实际像素面积（整数）
        "像素间距还原面积": [
            f"{int(ref_max_area)}" if f == "1.jpg" or f == "1.png" or f == "1.jpeg" else int(a)
            for f, a in zip(df['filename'], df['还原面积_pixel'].round(0))
        ],  # 像素间距法还原面积（参考图片为 ref_max_area）
        "像素间距还原面积差异": [
            "0.00%" if f == "1.jpg" or f == "1.png" or f == "1.jpeg" else f"{v:.2f}%"
            for f, v in zip(df['filename'], df['差异_pixel'].round(2))
        ],  # 像素间距法误差（参考图片为 0%）
        "姿态变化还原面积": [
            f"{int(ref_max_area)}" if f == "1.jpg" or f == "1.png" or f == "1.jpeg" else int(a)
            for f, a in zip(df['filename'], df['还原面积_pnp'].round(0))
        ],  # PnP 法还原面积
        "姿态变化还原面积差异": [
            "0.00%" if f == "1.jpg" or f == "1.png" or f == "1.jpeg" else f"{v:.2f}%"
            for f, v in zip(df['filename'], df['差异_pnp'].round(2))
        ],  # PnP 法误差
        "姿态变化+角度补偿还原面积": [
            f"{int(ref_max_area)}" if f == "1.jpg" or f == "1.png" or f == "1.jpeg" else int(a)
            for f, a in zip(df['filename'], df['还原面积_pnp_angle'].round(0))
        ],  # 姿态+角度补偿法还原面积
        "姿态变化+角度补偿还原面积差异": [
            "0.00%" if f == "1.jpg" or f == "1.png" or f == "1.jpeg" else f"{v:.2f}%"
            for f, v in zip(df['filename'], df['差异_pnp_angle'].round(2))
        ],  # 姿态+角度补偿法误差
    }

    df_transposed = pd.DataFrame(transpose_table)  # 创建转置表格，便于阅读和输出

    csv_path = f"{args.mode}_变化纵向还原面积对比.csv"  # 定义输出 CSV 文件路径

    # 检查 CSV 文件是否存在，若存在则追加数据并插入空行
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')  # 读取现有 CSV 文件
        blank_row = pd.DataFrame([[""] * len(df_transposed.columns)], columns=df_transposed.columns)  # 创建空行
        combined_df = pd.concat([existing_df, blank_row, df_transposed], ignore_index=True)  # 合并数据
        combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 保存合并后的 CSV
    else:
        df_transposed.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 直接保存新 CSV

    print(f"\n=== {args.mode}变化 纵向对比表 ===")  # 打印标题
    print(df_transposed.to_string(index=False))  # 打印表格内容