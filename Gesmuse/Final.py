import cv2  
import numpy as np
import math
import random
import os
from matplotlib import pyplot as plt
def transform(img, x0, x1, y0, y1, c):
	x = random.uniform(x0, x1)
	y = random.uniform(y0, y1)
	scale = 1
	degree = 0 #random.uniform(-20, 20)
	M1 = np.array([[1, 0, x], [0, 1, y]])
	M2 = cv2.getRotationMatrix2D((128, 128), degree, scale)
	img = cv2.warpAffine(img, M2, (256, 256))
	img = cv2.warpAffine(img, M1, (256, 256))
	result = cv2.resize(img, (256, 256))
	s = "G:/DataSet/train/" + str(c) + ".jpg"
	cv2.imwrite(s, img)
	cv2.destroyAllWindows()
c = 0
train_answer = open("G:/DataSet/train.txt",'w')
test_answer = open("G:/DataSet/test.txt",'w')
for i in range(32):
	for j in range(5):
		s = "G:/hands/hand/hand" + str(i) + "_" + str(j) + ".jpg"
		img = cv2.imread(s)
		x0, y0, w, h = cv2.boundingRect(img[:,:,0])
		x0 = -x0
		y0 = -y0
		x1 = 256 + x0 - w
		y1 = 256 + y0 - h
		for k in range(200):
			transform(img, x0, x1, y0, y1, c)
			c = c + 1
	s = "G:/hands/hand_test/hand" + str(i) + ".jpg"
	img = cv2.imread(s)
	s = "G:/DataSet/test/" + str(i) + ".jpg"
	cv2.imwrite(s, img)
a = [1,2,4,8,16]
for i in range(32):
	for k in range(1000):
		for j in range(5):
			train_answer.write(str(int(i & a[j] != 0)) + " ")
		train_answer.write("\n")
	for j in range(5):
		test_answer.write(str(int(i & a[j] != 0)) + " ")
	test_answer.write("\n")
train_answer.close()
test_answer.close()