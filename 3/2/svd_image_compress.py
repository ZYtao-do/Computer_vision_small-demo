from PIL import Image
import numpy as np
import cv2


a = Image.open("bf.bmp")
b = a.convert('L')
matrix = np.matrix(b)

#svd分解
s,v,d = np.linalg.svd(matrix)


num_val = 60
r = s[:,:num_val]*np.diag(v[:num_val])*d[:num_val,:]

r = r.astype(np.uint8)

cv2.imshow('pca_image_induction', r)
cv2.waitKey(0)
cv2.destroyAllWindows()
