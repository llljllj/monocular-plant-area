# camera_calibration.py
# 相机标定程序 - 使用棋盘格图像计算相机内参和畸变系数

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class CameraCalibrator:
    """
    相机标定类
    使用棋盘格图像进行相机标定，输出相机内参矩阵和畸变系数
    """
    
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        初始化标定器
        
        参数:
        - chessboard_size: tuple, 棋盘格内角点数量 (列数, 行数)
          注意: 这里是内角点数，不是格子数
          例如: 9x6 表示棋盘有 10x7=70 个格子，但只有 9x6=54 个内角点
        - square_size: float, 每个格子的物理尺寸 (单位: 毫米)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 准备物体点坐标 (棋盘格在世界坐标系中的3D坐标)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # 乘以实际尺寸
        
        # 存储所有图像的物体点和图像点
        self.objpoints = []  # 3D 点在真实世界空间
        self.imgpoints = []  # 2D 点在图像平面
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
    
    def find_corners(self, image_path, visualize=True):
        """
        在单张图像中查找棋盘格角点
        
        参数:
        - image_path: str, 图像路径
        - visualize: bool, 是否可视化显示角点检测结果
        
        返回:
        - success: bool, 是否成功找到角点
        - corners: ndarray, 角点坐标 (如果成功)
        """
        # 读取图像 (使用 imdecode 支持中文路径)
        try:
            img_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except:
            img = cv2.imread(image_path)
        
        if img is None:
            print(f"❌ 无法读取图像: {image_path}")
            return False, None
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 亚像素精细化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 可视化
            if visualize:
                img_vis = img.copy()
                cv2.drawChessboardCorners(img_vis, self.chessboard_size, corners_refined, ret)
                
                # 调整显示大小
                h, w = img_vis.shape[:2]
                scale = min(800 / w, 600 / h)
                if scale < 1:
                    img_vis = cv2.resize(img_vis, None, fx=scale, fy=scale)
                
                cv2.imshow('Chessboard Corners', img_vis)
                cv2.waitKey(300)  # 显示 300ms
            
            print(f"✅ 成功检测: {os.path.basename(image_path)}")
            return True, corners_refined
        else:
            print(f"❌ 未找到角点: {os.path.basename(image_path)}")
            return False, None
    
    def calibrate(self, images_folder, image_format='*.jpg', visualize=True, save_path='camera_params.npz'):
        """
        使用文件夹中的所有棋盘格图像进行相机标定
        
        参数:
        - images_folder: str, 包含标定图像的文件夹路径
        - image_format: str, 图像文件格式 (支持通配符)
        - visualize: bool, 是否显示角点检测过程
        - save_path: str, 标定结果保存路径 (.npz 文件)
        
        返回:
        - success: bool, 标定是否成功
        """
        # 获取所有图像路径
        image_paths = glob.glob(os.path.join(images_folder, image_format))
        
        # 支持多种图像格式
        if len(image_paths) == 0:
            for ext in ['*.png', '*.jpeg', '*.bmp', '*.tiff']:
                image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
        
        if len(image_paths) == 0:
            print(f"❌ 错误: 在文件夹 '{images_folder}' 中未找到图像文件")
            return False
        
        print(f"\n📷 开始相机标定...")
        print(f"棋盘格规格: {self.chessboard_size[0]}x{self.chessboard_size[1]} 内角点")
        print(f"格子尺寸: {self.square_size} mm")
        print(f"找到 {len(image_paths)} 张图像\n")
        
        # 处理每张图像
        valid_images = 0
        img_size = None
        
        for img_path in image_paths:
            success, corners = self.find_corners(img_path, visualize=visualize)
            
            if success:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                valid_images += 1
                
                # 获取图像尺寸（支持中文路径）
                if img_size is None:
                    try:
                        img_data = np.fromfile(img_path, dtype=np.uint8)
                        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    except:
                        img = cv2.imread(img_path)
                    
                    if img is not None:
                        h, w = img.shape[:2]
                        img_size = (w, h)
        
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"有效图像数: {valid_images} / {len(image_paths)}")
        
        if valid_images < 3:
            print(f"❌ 错误: 有效图像太少 (至少需要 3 张)，标定失败")
            return False
        
        # 执行标定
        print(f"\n🔧 正在计算相机参数...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            img_size,
            None, 
            None
        )
        
        if not ret:
            print(f"❌ 标定失败")
            return False
        
        # 保存结果
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # 计算重投影误差
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                rvecs[i], 
                tvecs[i], 
                mtx, 
                dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        self.calibration_error = total_error / len(self.objpoints)
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"✅ 标定成功！")
        print(f"{'='*60}\n")
        
        print(f"📊 相机内参矩阵 (Camera Matrix):")
        print(f"   fx = {mtx[0, 0]:.2f}  (焦距 x)")
        print(f"   fy = {mtx[1, 1]:.2f}  (焦距 y)")
        print(f"   cx = {mtx[0, 2]:.2f}  (主点 x)")
        print(f"   cy = {mtx[1, 2]:.2f}  (主点 y)")
        print(f"\n{mtx}\n")
        
        print(f"📐 畸变系数 (Distortion Coefficients):")
        print(f"   k1 = {dist[0, 0]:.6f}  (径向畸变)")
        print(f"   k2 = {dist[0, 1]:.6f}  (径向畸变)")
        print(f"   p1 = {dist[0, 2]:.6f}  (切向畸变)")
        print(f"   p2 = {dist[0, 3]:.6f}  (切向畸变)")
        print(f"   k3 = {dist[0, 4]:.6f}  (径向畸变)")
        print(f"\n{dist}\n")
        
        print(f"📏 平均重投影误差: {self.calibration_error:.4f} 像素")
        print(f"   (误差越小越好，通常 < 0.5 为优秀)")
        
        # 保存到 npz 文件
        np.savez(
            save_path,
            camera_matrix=mtx,
            dist_coeffs=dist,
            rvecs=rvecs,
            tvecs=tvecs,
            calibration_error=self.calibration_error,
            image_size=img_size,
            chessboard_size=self.chessboard_size,
            square_size=self.square_size
        )
        
        print(f"\n💾 标定结果已保存到: {save_path}")
        print(f"{'='*60}\n")
        
        return True
    
    def test_undistortion(self, test_image_path, save_result=True):
        """
        测试畸变校正效果
        
        参数:
        - test_image_path: str, 测试图像路径
        - save_result: bool, 是否保存校正后的图像
        """
        if self.camera_matrix is None:
            print("❌ 错误: 请先进行标定")
            return
        
        # 读取图像 (支持中文路径)
        try:
            img_data = np.fromfile(test_image_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except:
            img = cv2.imread(test_image_path)
        
        if img is None:
            print(f"❌ 无法读取图像: {test_image_path}")
            return
        
        h, w = img.shape[:2]
        
        # 获取最优新相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, 
            self.dist_coeffs, 
            (w, h), 
            1, 
            (w, h)
        )
        
        # 畸变校正
        dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # 裁剪图像
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # 显示对比
        img_compare = np.hstack([
            cv2.resize(img, (640, 480)),
            cv2.resize(dst, (640, 480))
        ])
        
        cv2.imshow('Original (Left) vs Undistorted (Right)', img_compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果 (支持中文路径)
        if save_result:
            output_path = test_image_path.replace('.', '_undistorted.')
            try:
                # 使用 imencode 支持中文路径
                is_success, buffer = cv2.imencode('.jpg', dst)
                if is_success:
                    buffer.tofile(output_path)
                    print(f"✅ 校正后的图像已保存到: {output_path}")
            except:
                cv2.imwrite(output_path, dst)
                print(f"✅ 校正后的图像已保存到: {output_path}")


def load_calibration(npz_path):
    """
    加载已保存的标定结果
    
    参数:
    - npz_path: str, .npz 文件路径
    
    返回:
    - camera_matrix: ndarray, 相机内参矩阵
    - dist_coeffs: ndarray, 畸变系数
    """
    data = np.load(npz_path)
    
    print(f"\n📂 加载标定文件: {npz_path}")
    print(f"{'='*60}")
    print(f"相机内参矩阵:\n{data['camera_matrix']}\n")
    print(f"畸变系数:\n{data['dist_coeffs']}\n")
    print(f"标定误差: {data['calibration_error']:.4f} 像素")
    print(f"图像尺寸: {data['image_size']}")
    print(f"{'='*60}\n")
    
    return data['camera_matrix'], data['dist_coeffs']


# ========== 主程序 ==========
if __name__ == "__main__":
    """
    使用说明:
    
    1. 准备棋盘格图像:
       - 打印一个 10x7 格子的棋盘格 (9x6 内角点)
       - 每个格子 25mm × 25mm
       - 固定在平整硬板上
    
    2. 拍摄标定图像 (15-30 张):
       - 从不同角度拍摄 (正面、左右倾斜、上下倾斜)
       - 覆盖图像的不同区域 (中心、边角)
       - 保持棋盘格清晰、完整可见
       - 保存到 calibration_images/ 文件夹
    
    3. 运行标定:
       python camera_calibration.py
    """
    
    # ========== 配置参数 ==========
    # 棋盘格规格 (内角点数量)
    CHESSBOARD_SIZE = (9, 6)  # 列数 × 行数 (9x6 内角点 = 10x7 格子)
    
    # 每个格子的物理尺寸 (毫米)
    SQUARE_SIZE = 25.0  # 25mm = 2.5cm
    
    # 标定图像文件夹
    IMAGES_FOLDER = r"D:\植物面积还原算法\calibration_images"
    
    # 输出文件路径
    OUTPUT_FILE = r"D:\植物面积还原算法\camera_params.npz"
    
    # 是否显示角点检测过程
    VISUALIZE = True
    # ==============================
    
    # 检查文件夹是否存在
    if not os.path.exists(IMAGES_FOLDER):
        print(f"❌ 错误: 标定图像文件夹不存在: {IMAGES_FOLDER}")
        print(f"请创建该文件夹并放入棋盘格标定图像")
        exit(1)
    
    # 创建标定器
    calibrator = CameraCalibrator(
        chessboard_size=CHESSBOARD_SIZE,
        square_size=SQUARE_SIZE
    )
    
    # 执行标定
    success = calibrator.calibrate(
        images_folder=IMAGES_FOLDER,
        visualize=VISUALIZE,
        save_path=OUTPUT_FILE
    )
    
    if success:
        print(f"🎉 标定完成！现在可以在 pnp.py 中使用标定结果：")
        print(f"\n# 在 pnp.py 中添加以下代码:")
        print(f"data = np.load('{OUTPUT_FILE}')")
        print(f"camera_matrix = data['camera_matrix']")
        print(f"dist_coeffs = data['dist_coeffs']")
        
        # 询问是否测试畸变校正
        test_images = glob.glob(os.path.join(IMAGES_FOLDER, '*.jpg'))
        if len(test_images) > 0:
            print(f"\n是否要测试畸变校正效果？(按 Enter 跳过，输入 y 测试)")
            user_input = input().strip().lower()
            if user_input == 'y':
                calibrator.test_undistortion(test_images[0])
    else:
        print(f"\n💡 标定失败，请检查:")
        print(f"   1. 棋盘格规格是否正确 (当前设置: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} 内角点)")
        print(f"   2. 图像是否清晰，棋盘格完整可见")
        print(f"   3. 至少需要 3 张有效图像 (建议 15-30 张)")
