import os
import cv2
import numpy as np 
import scipy as sp


def getImages(imageDir,nS):
	dataSet = np.ndarray(shape = (800,nS,nS),dtype = np.float32)
	i = 0
	for fileName in sorted(os.listdir(imageDir)):
		file = imageDir + fileName
		img = cv2.imread(file,0)
		reSized = cv2.resize(img,(nS,nS),interpolation = cv2.INTER_NEAREST)
		dataSet[i] = reSized
		i += 1
	x_train = np.reshape(dataSet,(len(dataSet),nS,nS,1))
	return x_train/255.0

def getLabels(labelDir):
		labels = []
		for fileName in sorted(os.listdir(labelDir)):
			file = labelDir + fileName
			with open(file,'r') as f:
				lines = f.readlines()
				if 'normal' in lines:
					labels.append(0.0)
				else:
					labels.append(1.0)
				f.close()
		return np.array(labels)

def getTestData(testDir,nS):
	dataSet = np.ndarray(shape = (96,nS,nS),dtype = np.float32)
	j = 0
	for fileName in sorted(os.listdir(testDir)):
		file = testDir + fileName 
		img = cv2.imread(file,0)
		reSized = cv2.resize(img,(nS,nS),interpolation = cv2.INTER_NEAREST)
		dataSet[j] = reSized
		j += 1
	x_test = np.reshape(dataSet,(96,nS,nS,1))
	return x_test


