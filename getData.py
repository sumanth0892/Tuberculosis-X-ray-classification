import os
import cv2
import numpy as np 
import scipy as sp


def trainingData(imageDir,w):
	dataSet = np.ndarray(shape = (len(os.listdir(imageDir))//3,512,512,1),dtype = np.float32)
	masks = np.ndarray(shape = (len(os.listdir(imageDir))//3,512,512,1),dtype = np.float32)
	i = 0; labels = []; j = 0
	for fileName in sorted(os.listdir(imageDir)):
		if fileName == '.DS_Store' or 'mask' in fileName:
			continue
		if '.txt' in fileName:
			if '_0.txt' in fileName:
				labels.append(0.0)
			if '_1.txt' in fileName:
				labels.append(1.0)
		if '.txt' not in fileName:
			img = cv2.imread(imageDir + fileName,0)
			reSized = cv2.resize(img,(w,w),interpolation = cv2.INTER_AREA)
			dataSet[i] = reSized
		i += 1
	for fileName in sorted(os.listdir(imageDir)):
		if 'mask' in fileName:
			img = cv2.imread(imageDir + fileName,0)
			reSized = cv2.resize(img,(w,w),interpolation = cv2.INTER_AREA)
			masks[i] = reSized
			j += 1
		if 'mask' not in fileName:
			continue 
	return dataSet,masks,np.array(labels)


def getTestData(testDir,w):
	dataSet = np.ndarray(shape = (len(os.listdir(imageDir))//2,w,w,1),dtype = np.float32)
	k = 0; labels = []
	for fileName in sorted(os.listdir(testDir)):
		if fileName == '.DS_Store':
			continue
		if '.txt' in fileName:
			if '_0.txt' in fileName:
				labels.append(0.0)
			if '_1.txt' in fileName:
				labels.append(1.0)
		if '.txt' not in fileName:
			img = cv2.imread(testDir + fileName,0)
			reSized = cv2.resize(img,(w,w),interpolation = cv2.INTER_AREA)
			dataSet[k] = reSized
			k += 1
	return dataSet



