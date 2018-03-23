# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:51:10 2018

@author: Vashi NSIT
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import imutils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

def image_to_feature_vector(image, size=(120,160)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image,size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()

#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset")
#ap.add_argument("-k", "--neighbors", type=int, default=1,
#	help="# of nearest neighbors for classification")
#ap.add_argument("-j", "--jobs", type=int, default=-1,
#	help="# of jobs for k-NN distance (-1 uses all available cores)")
#args = vars(ap.parse_args())


#C:/Users/Vashi NSIT/Desktop/Data Science/Kaggle/DogsvsCats/train

# grab the list of images that we'll be describing
#print("[INFO] describing images...")
pathnames=(os.listdir("TrainDirectory"))
path=[x[0] for x in os.walk("TrainDirectory")]
path[1]
def definepaths():
    pathnames=(os.listdir("TrainDirectory"))
    path=[x[0] for x in os.walk("TrainDirectory")]
    i=1
    patternpath=dict()
    for name in pathnames:
        print(name)
        print(path[i])
        #globals()[name]=path[i]
        patternpath[name]=list(paths.list_images(path[i]))
        i=i+1
    return patternpath


patternpath=definepaths()
patternpath['abstract']

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list

# loop over the input images
tol=[]
for key in patternpath:
    print(key)
    i=patternpath[key]
    print(type(i))
    for k in i:
        im=cv2.imread(k)
        print(type(im))
        tol.append(key)
        break
    break
tol
im.shape

im[:,:,1].shape
imresize=cv2.resize(im,(60,80))
imflat=imresize.flatten()
imflat.shape
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
type(imhsv)
imhsv.shape
imhist = cv2.calcHist([imhsv], [0, 1, 2], None, (8,8,8),[0, 180, 0, 256, 0, 256])
imhist.shape
imhist


def image_vector(patternpath): 
    rawImages = []
    features = []
    labels = []
    for key in patternpath:
        value=patternpath[key]
        
        for (i, imagePath) in enumerate(value):
            
            # load the image and extract the class label (assuming that our
            # path as the format: /path/to/dataset/{class}.{image_num}.jpg
            image = cv2.imread(imagePath)
            # extract raw pixel intensity "features", followed by a color
            # histogram to characterize the color distribution of the pixels
            # in the image
            pixels = image_to_feature_vector(image)
            hist = extract_color_histogram(image)
            # update the raw images, features, and labels matricies,
            # respectively
            rawImages.append(pixels)
            features.append(hist)
            labels.append(key)
    return rawImages,features,labels#,image,pixels,hist
     
rawImages,features,labels=image_vector(patternpath)             
len(labels)
       
# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))


#For Neural Networks use this
from sklearn import preprocessing  
enc=preprocessing.LabelEncoder()
labels_cat=enc.fit_transform(labels)
labels_cat=labels_cat.reshape(labels.shape[0],1)
labels_cat.shape
encoder = preprocessing.OneHotEncoder(sparse=False)  
labels_onehot = encoder.fit_transform(labels_cat)  
labels_onehot.shape

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels_onehot, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels_onehot, test_size=0.25, random_state=42)

model=MLPClassifier(solver='lbfgs',hidden_layer_sizes=(20,8),activation='logistic')
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))
#very poor accuracy=0%
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc * 100))
#accuracy=8%

# For SVM and KNN use this

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)


model = KNeighborsClassifier(n_neighbors=5,
	n_jobs=-1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))
#good accuracy than other =24%

model = SVC(C=30,gamma=0.01)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))
#Accuracy=8%

# train and evaluate a k-NN classifer on the histogram
# representations

model = KNeighborsClassifier(n_neighbors=3,
	n_jobs=-1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc * 100))
# Best of all accuracy=28%

model = SVC(C=30,gamma=0.01)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc * 100))
#accuracy=19% 

