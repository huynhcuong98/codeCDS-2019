    
import glob
import time
import cv2
from collections import deque
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn import svm
import joblib

def train_SVC(X_train, y_train):
    """
        Function to train an svm.
    """
    svc = svm.LinearSVC( C=0.005,max_iter=10000)
    # svc=svm.SVC(kernel = "rbf" , gamma = 10, coef0 = 0)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc
class train_1ob(object):
    def __init__(self, pos, neg, n_sample):
        self.pos = pos
        self.neg = neg 
        self.n_sample = n_sample
        self.folder = self.pos, self.neg
        print(self.folder)
    def setup_train_data(self):
        X_train = []
        y_train = []

        for i in range(len(self.folder)):
            listitem=[]
            for item in glob.glob(f'{self.folder[i]}/*.jpg'):
                listitem.append(item)
            print(f"{self.folder[i]}:"+str(len(listitem)))
            for j in range(self.n_sample):
                img = cv2.imread("{}".format(listitem[j]))           #40x40x3
                # dim = np.random.randint(15,24)
                dim = 40
                dims=(dim,dim)
                img = cv2.resize(img, dims)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fd, hog_image = hog(img, orientations = 9, pixels_per_cell= (5,5), cells_per_block = (8,8), visualize = True, multichannel = True)
                # print(len(fd))
                y_train = np.append(y_train,f"{self.folder[i]}")
                X_train = np.append(X_train,fd)  
        # print(len(listitem))
        return X_train, y_train
    print('Preparing training data...')
    def main(self):
        X_train, y_train = self.setup_train_data()
        X_train= X_train.reshape(-1,len(fd)).astype('float32')
        # print(X_train.shape) 
        # print(y_train.shape) 
        svc = train_SVC(X_train, y_train)
        filename = f'model/{self.pos}.xml'
        joblib.dump(svc, filename, protocol= 2)
class train_multi_ob(object):
    def __init__(self, n_class, n_sample):
        # self.n_sample = n_sample
        self.folder = n_class
        print(self.folder)
    def setup_train_data(self):
        X_train = []
        y_train = []

        for i in range(len(self.folder)):
            listitem=[]
            for item in glob.glob(f'{self.folder[i]}/*.jpg'):
                listitem.append(item)
            print(f"{self.folder[i]}:"+str(len(listitem)))
            for j in range(len(listitem)):
                img = cv2.imread("{}".format(listitem[j]))           #40x40x3
                # dim = np.random.randint(15,24)
                self.dim = 40
                dims=(self.dim,self.dim)
                img = cv2.resize(img, dims)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.fd, hog_image = hog(img, orientations = 9, pixels_per_cell= (5,5), cells_per_block = (8,8), visualize = True, multichannel = True)
                # print(len(fd))
                y_train = np.append(y_train,f"{self.folder[i]}")
                X_train = np.append(X_train,self.fd)  
        # print(len(listitem))
        return X_train, y_train
    print('Preparing training data...')

    def main(self):
        X_train, y_train = self.setup_train_data()
        X_train= X_train.reshape(-1,len(self.fd)).astype('float32')
        # print(X_train.shape) 
        # print(y_train.shape) 
        svc = train_SVC(X_train, y_train)
        filename = f'model/model_{self.dim}.xml'
        joblib.dump(svc, filename, protocol= 2)

def model():
    print('training...')
    n_class = 're','cam-re', 'straight','stop','none'
    # n_class = 'stop','straight'
    model= train_multi_ob(n_class=n_class,n_sample=1220)
    model.main()
model()
