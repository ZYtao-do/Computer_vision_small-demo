##自适应求K值
import numpy as np
import cv2 as cv
'''
（animals）分别求每个维度的平均值，然后对于所有的样例，都减去对应维度的均值，得到去中心化的数据；

（food）求协方差矩阵C：用去中心化的数据矩阵乘上它的转置，然后除以(N-animals)即可，N为样本数量；

（scenery）求协方差的特征值和特征向量；

（4）将特征值按照从大到小排序，选择前k个，然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵；

（5）将样本点从原来维度投影到选取的k个特征向量，得到低维数据；

（6）通过逆变换，重构低维数据，进行复原。

'''

# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    k = Percentage2n(D, p)  # 确定k值
    print("保留99%信息，降维后的特征个数：" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA算法
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)
    # 重构数据
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat


def main():
    imagePath = 'photo4.png'
    image = cv.imread(imagePath)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rows, cols = image.shape
    print("降维前的特征个数：" + str(cols) + "\n")
    print(image)
    print('----------------------------------------')
    reconImage = PCA(image, 0.6) # 通过改变保留信息的程度来看这个图片的特征值 
    reconImage = reconImage.astype(np.uint8)
    print(reconImage)
    cv.imshow('test_pic', reconImage)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
