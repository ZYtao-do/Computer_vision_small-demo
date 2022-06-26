import numpy as np
import cv2 as cv


# 数据中心化
def center_image(pre_image):
    rows, cols = pre_image.shape
    mean_image = np.mean(pre_image, axis=0)
    mean_image = np.tile(mean_image, (rows, 1))

    newdata = pre_image - mean_image
    return newdata, mean_image


def tz(cov_image):
    va, ve = np.linalg.eig(cov_image)  # 得到特征值和特征向量
    k = 60
    eigenvalue = np.argsort(va)
    tz_value = eigenvalue[-1:-k:-1]
    tz_vector = ve[:, tz_value]
    return tz_value, tz_vector


# 重构图像
def Reconstruction(pre_image, tz_vector, mean_image):
    recon_image = pre_image * tz_vector * tz_vector.T + mean_image
    return recon_image


# PCA算法
def PCA(data):
    pre_image = np.float32(np.mat(data))
    # 数据中心化
    pre_image, mean_image = center_image(pre_image)
    # 计算协方差矩阵
    cov_image = np.cov(pre_image, rowvar=0)
    # 得到k个特征值和特征向量
    D, V = tz(cov_image)
    # 重构数据
    recon_image = Reconstruction(pre_image, V, mean_image)
    return recon_image


def main():
    image = cv.imread("bf.bmp")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    reconImage = PCA(image)
    reconImage = reconImage.astype(np.uint8)
    print(reconImage)

    cv.imshow('pca_image_induction', reconImage)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()