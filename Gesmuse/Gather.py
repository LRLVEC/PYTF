#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' try12-rectangle&voi '

__author__ = 'Adam'
PD=60
import numpy as np
import cv2
nn=0
mm=0
FRAME_RAW=360
FRAME_COL=400
WINDOW_X=200
WINDOW_Y=200
zero=np.zeros((256,256,3),np.int32)
img=np.zeros((256,256,1),np.uint8)
'''
cv2.rectangle(img,(0,0),(128,128),255,3)
cv2.imshow('img',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()
'''
'''
if __name__=='__main__':
	cap=cv2.VideoCapture(0)
	cap_raw=cap.get(3)
	cap_col=cap.get(4)
	cv2.namedWindow('frame')
	cv2.resizeWindow('frame',1000,2000)
	while(1):
		ret,frame=cap.read()
		cv2.rectangle(frame,(0,0),(255,255),(255,255,255),3)
		cv2.imshow('frame',frame)
		k=cv2.waitKey(5)&0xff
		if k==27:
			break
	cv2.destroyAllWindows()
	cap.release()
'''
if __name__=='__main__':
	x=30
	y=100
	cap=cv2.VideoCapture(0)
	cv2.namedWindow('frame')
	cv2.moveWindow('frame',WINDOW_Y,WINDOW_X)
	
	while(1):
		ret,frame=cap.read()
		frame2=frame.copy()
		cv2.rectangle(frame2,(0,PD),(FRAME_COL-1,FRAME_RAW+PD-1),(255,255,255),3)
		cv2.imshow('frame',frame2)
		k=cv2.waitKey(5)&0xff
		if k==27:
			print('succeed!')
			break
	cv2.destroyAllWindows()
	
	
	voi2=frame[PD:FRAME_COL+PD,0:FRAME_RAW].copy().astype(np.int32)
	#取反
	#voi_not=cv2.bitwise_not(voi2)
	cv2.namedWindow('voi')
	cv2.moveWindow('voi',WINDOW_Y,WINDOW_X)
	while(1):
		ret,frame=cap.read()
		voi=frame[PD:FRAME_COL+PD,0:FRAME_RAW].astype(np.int32)
		#做差
		sub=np.abs(voi-voi2).astype(np.uint8)
		#灰度化
		sub_gray=cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY)
		#二值化
		ret1,img=cv2.threshold(sub_gray,x,y,cv2.THRESH_BINARY)#???

		kernel = np.ones((5,5),np.uint8)
		opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		#ct1=cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
		ct2=cv2.Canny(closing,100,200)#???
		ct3, hierarchy = cv2.findContours(ct2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		#新建图层
		img2=img.copy()
		img2[:,:]=0
		#挑选主体图层

		#cv2.imwrite('hand_result.jpg',img2)
		#print(len(ct3))
		l=([len(ct3[i]) for i in range(len(ct3))])
		if l:
			n=l.index(max(l))
			cv2.drawContours(img2,ct3,n,(255,255,255),2)
		#print(n)
		
		cv2.imshow('voi',img2)
		k=cv2.waitKey(5)&0xff
		if k==ord('q'):
			cv2.imwrite('./hands/hand_enhance/%d_%d.png'%(nn,mm),img2)
			mm+=1
			#if mm==4:
			#	nn+=1
			#	mm=0
			#else:
			#	mm+=1
		elif k==ord('a'):
			nn+=1
		elif k==27:
			break
	
	cv2.destroyAllWindows()
	print(x,'  ',y)
	cap.release()
