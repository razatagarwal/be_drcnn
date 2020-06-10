#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np 

def preprocess_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('21.png', image)
	image = cv2.bitwise_not(image)
	kernel = np.ones((30, 30),np.uint8)
	tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
	cv2.imwrite('22.png', tophat)
	
	g_kernel = cv2.getGaborKernel((350, 350), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
	h, w = g_kernel.shape[:2]
	g_kernel = cv2.resize(tophat, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
	image = cv2.bitwise_not(g_kernel)
	cv2.imwrite('23.png', image)
	
	for i in range(len(image)):
		for j in range(len(image)):
			if image[i][j] > 244:
				image[i][j] = 0
	cv2.imwrite('24.png', tophat)
	kernel = np.ones((2, 2),np.uint8)
	opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	cv2.imwrite('25.png', tophat)
	return opening