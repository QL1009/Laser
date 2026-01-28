import cv2
import numpy as np
import os


# 棋盘格内角点行列数
pattern_size = (9, 6)

# 假设棋盘格每个方格边长为1（实际应用中需根据真实尺寸调整,单位：米）
square_size = 1

# 准备棋盘格角点的世界坐标
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

objpoints = []  # 用于存储世界坐标点
imgpoints = []  # 用于存储图像坐标点

# 标定板照片所在文件夹
images_folder = 'H:\PyPrj\Laser\Camera__Calibration\data'
images = os.listdir(images_folder)
for fname in images:
    if fname.endswith('.jpg') or fname.endswith('.png'):
        img = cv2.imread(os.path.join(images_folder, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            # 亚像素级角点优化
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners)
            # 绘制并显示角点
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("相机内参矩阵：\n", mtx)
print("畸变系数：\n", dist)
