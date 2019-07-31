import cv2
import numpy as np
import math
import random
import os
import time

all_size = 2755
width = 360
height = 400
origin = np.empty([all_size, height, width], dtype = np.uint8)
answer0 = np.empty([all_size, 5], dtype = np.float32)
answer1 = np.empty([all_size, 5], dtype = np.bool)
M1 = np.array([[1,0,width - 0.5 * width],[0,1,height - 0.5 * height]])

def init():
	a = 0
	b = 0
	kk = np.array([1,2,4,8,16])
	while(1):
		s = "L:/Temp/hand_enhance/" + str(a) + "_" + str(b) + ".png"
		if os.path.exists(s):
			origin[b] = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
			answer1[b] = np.bitwise_and(kk, a).astype(np.bool)
			b += 1
		else:
			a += 1
			if a > 31:break
	global answer0
	answer0 = answer1.astype(np.float32) * 2.0 - 1.0
def run(num):
	final = np.empty([num, 256, 256], dtype = np.float32)
	id = np.random.choice(all_size, num)
	for i in range(num):
		degree = random.gauss(0, 10)
		if abs(degree) > 30:degree = np.sign(degree) * 30
		M = cv2.getRotationMatrix2D((width / 2, height / 2), degree,1)
		img = cv2.warpAffine(cv2.warpAffine(origin[id[i]], M,(2 * width,2 * height)), M1,(2 * width,2 * height))
		ret, thresh = cv2.threshold(img.copy(),127, 255, cv2.THRESH_BINARY)
		contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		box = np.empty([4,2])
		for c in contours:
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
		dm = box[3] - box[0]
		dn = box[1] - box[0]
		m = math.sqrt(np.sum(dm * dm))
		n = math.sqrt(np.sum(dn * dn))
		if m > n:
			second_lowest = box[1][1]
			if 0.5 * m >= n:
				box[0] = (box[3] + box[0]) / 2
				box[1] = (box[2] + box[1]) / 2
			else:
				box[0] = box[3] - dm * (n / m)
				box[1] = box[2] - dm * (n / m)
		else:
			second_lowest = box[3][1]
			if 0.5 * n >= m:
				box[0] = (box[1] + box[0]) / 2
				box[3] = (box[3] + box[2]) / 2
			else:
				box[0] = box[1] - dn * (m / n)
				box[3] = box[2] - dn * (m / n)
		dd = box[2] - box[0]
		x1 = int(min(box[0][0],box[1][0],box[2][0],box[3][0])) - 1
		x2 = max(box[0][0],box[1][0],box[2][0],box[3][0]) + 1
		y1 = int(min(box[0][1],box[1][1],box[2][1],box[3][1])) - 1
		y2 = max(box[0][1],box[1][1],box[2][1],box[3][1]) + 1
		dy = second_lowest - y1
		r = math.sqrt(np.sum(dd * dd))
		if dy > r:r = int(random.uniform(r, dy))
		else:r = int(r)
		x3 = int(x2 - r)
		y3 = int(y2 - r)
		xx,yy = int(random.uniform(x3, x1)),int(random.uniform(y3, y1))
		final[i] = cv2.resize(img[yy:yy + r, xx:xx + r], (256,256))
		#cv2.imshow("contours2", final[i])
		#cv2.waitKey(1)
		#cv2.destroyAllWindows()
	return np.reshape(final,[num, 256, 256, 1]), answer0[id], answer1[id]
#init()
#bg = time.time()
#ahh, bhh, chh = run(32)
#ed = time.time()
#print(ed - bg)
#print(ahh)
#print(bhh)
#print(chh)