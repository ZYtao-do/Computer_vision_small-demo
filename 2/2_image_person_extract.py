import math

import cv2
from PIL import Image
from matplotlib import pyplot as plt


img1 = cv2.imread('./dataset/1.jpg', 0)
img2 = cv2.imread('./dataset/2.jpg', 0)
img3 = cv2.imread('./dataset/3.jpg', 0)
img4 = cv2.imread('./dataset/4.jpg', 0)


# 先进行高斯滤波，再Otus
blur1 = cv2.GaussianBlur(img1, (5, 5), 0)
ret1, th1 = cv2.threshold(blur1, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


blur2 = cv2.GaussianBlur(img2, (5, 5), 0)
ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

blur3 = cv2.GaussianBlur(img3, (5, 5), 0)
ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

blur4 = cv2.GaussianBlur(img4, (5, 5), 0)
ret4, th4 = cv2.threshold(blur4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img1 = Image.open('./dataset/1.jpg')
img2 = Image.open('./dataset/2.jpg')
img3 = Image.open('./dataset/3.jpg')
img4 = Image.open('./dataset/4.jpg')

images = [img1, th1, img2, th2, img3, th3, img4, th4]
titles = ['image1', "working1", 'image2', "working2", 'image3', "working3", 'image4', "working4"]


for i in range(4):
    # 绘制原图
    plt.subplot(4, 2, i * 2 + 1)
    plt.imshow(images[i * 2])
    plt.title(titles[i * 2], fontsize=8)
    plt.xticks([])
    plt.yticks([])

    # 绘制阈值图
    plt.subplot(4, 2, i * 2 + 2)
    plt.imshow(images[i * 2 + 1], 'gray')
    plt.title(titles[i * 2 + 1], fontsize=8)
    plt.xticks([])
    plt.yticks([])


plt.show()


def otus(image1):
    m = image1.mean()
    hist = cv2.calcHist([image1],
                        [0],  # 使用的通道
                        None,  # 没有使用mask
                        [255],  # HistSize
                        [0, 255])  # 直方图柱的范围

    all_variance = []
    for th in range(0, 256):
        pixel_prob = hist / image1.size
        mu0 = hist[:th].mean()
        mu1 = hist[th:].mean()
        w0_th = pixel_prob[:th].sum()
        w1_th = 1 - w0_th

        all_variance.append(w0_th * math.pow(mu0 - m, 2) + w1_th * math.pow(mu1 - m, 2))
    print("threshold: ", all_variance.index(max(all_variance)))

    return all_variance.index(max(all_variance))

