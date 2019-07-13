import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

img = cv2.imread("ahh.jpg",cv2.IMREAD_COLOR)
print(img)
#method1
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.imshow(img2)
plt.show()

#method2
img3 = img[:,:,::-1]
plt.imshow(img3)
plt.show()

#method3
img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img4)
plt.show()
