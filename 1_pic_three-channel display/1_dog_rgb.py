import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
im = cv2.imread('dog.jpg')
print('the data demension of input image is:')
print(im.shape)
plt.subplot(221)
plt.title('org image')
plt.imshow(im)

b,g,r = cv2.split(im)


# 显示图片的第一个通道
dog_Rc = np.zeros((576,768,3),np.uint8)
dog_Rc[:, :, 0] = b
plt.subplot(222)
plt.title('Red channel')
plt.imshow(dog_Rc)



# 显示图片的第二个通道
dog_Gc = np.zeros((576,768,3),np.uint8)
dog_Gc[:, :, 1] = g
plt.subplot(223)
plt.title('Green channel')
plt.imshow(dog_Gc)



# 显示图片的第三个通道
dog_Bc = np.zeros((576,768,3),np.uint8)
dog_Bc[:, :, 2] = b
plt.subplot(224)
plt.title('Blue channel')
plt.imshow(dog_Bc,cmap='Blues')
plt.show()
