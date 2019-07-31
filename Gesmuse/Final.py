import cv2
import numpy as np
import math 
import random

#for a in range(32):
#	img=cv2.imread("G:/DataSet/test/"+str(a)+".jpg")
#	img=img[:,::-1,:]
#	cv2.imwrite("G:/DataSet/test/"+str(a)+".jpg",img)
k = 0
for a in range(32):
	for b in range(5):
		for degree in np.random.normal(0, 10, 200):
			if abs(degree) > 30:degree = np.sign(degree) * 30
			s = "hands/hand_train/" + str(a) + "_" + str(b) + ".png"
			degree_ = 20 * math.exp(-0.1 * a)
			img = cv2.imread(s)
			cols, rows = img.shape[:2]
			#degree = random.uniform(-30,30)
			M = cv2.getRotationMatrix2D((cols / 2,rows / 2),degree + degree_,1)
			img = cv2.warpAffine(img,M,(2 * cols,2 * rows)) 
			M1 = np.array([[1,0,cols - 0.5 * cols],[0,1,rows - 0.5 * rows]])
			img = cv2.warpAffine(img,M1,(2 * cols,2 * rows))
			ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
			contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			for c in contours:
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
			#cv2.drawContours(img, [np.int32(box)], 0, (0, 0, 255), 3)
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
			if dy > r: 
				r = int(random.uniform(r, dy))
			else:
				r = int(r)
			x3 = int(x2 - r)
			y3 = int(y2 - r)
			#cv2.rectangle(img, (x1, y1), (int(x2), int(y2)), (0, 255, 0), 2)
			xx,yy = int(random.uniform(x3, x1)),int(random.uniform(y3, y1))
			#cv2.drawContours(img, [np.int32(box)], 0, (0, 0, 255), 3)
			#cv2.imshow("contours3", img)
			img = img[yy:yy + r, xx:xx + r]
			#cv2.imshow("contours", img)
			img = cv2.resize(img,(256,256))
			#cv2.imshow("contours2", img)
			s2 = "G:/DataSet/train/" + str(k) + ".png"
			k = k + 1
			cv2.imwrite(s2,img)
			#cv2.waitKey(1)
			cv2.destroyAllWindows()