#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, timeit
import pandas as pd
import numpy as np
from mllib import svm, tools, nn, optim, loss

X_train = pd.read_csv('Xtr.csv', header=None).as_matrix()[:, 0:-1].reshape(-1, 3, 32, 32)
X_test = pd.read_csv('Xte.csv', header=None).as_matrix()[:, 0:-1].reshape(-1, 3, 32 ,32)
Y_train = pd.read_csv('Ytr.csv').as_matrix()[:,1]

#################################
###      MODEL DEFINITION     ###
#################################

class MyNet(nn.Module):
    def __init__(self, depth_conv2=16):
        super().__init__()
        self.depth_conv2 = depth_conv2
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, depth_conv2, 5),
            nn.BatchNorm2d(depth_conv2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x.reshape(x.shape[0],-1)

    def backward(self, output_grad):
        output_grad = self.classifier.backward(output_grad)
        output_grad = self.flatten.backward(output_grad)
        return self.features.backward(output_grad)    

    def step(self, optimizer):
        self.classifier.step(optimizer)
        self.features.step(optimizer)

    def zero_grad(self):
        self.classifier.zero_grad()
        self.features.zero_grad()

    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

print('CNN training')
mynet = MyNet()
optimizer = optim.RMSprop(lr=0.001,lambda_reg=5.0)
criterion = loss.CrossEntropyLoss()

N = X_train.shape[0]
batch_size = 16
nb_batchs = int(N / batch_size)
nb_iterations = 20

start_global = timeit.default_timer()
optimizer._reset_state()
for epoch in range(0, nb_iterations, 1): # loop over the dataset multiple times
    running_loss = 0.0
    start = timeit.default_timer()
    suffle = np.random.permutation(N)
    X_train = X_train[suffle,:]
    Y_train = Y_train[suffle]
    for i in range(nb_batchs):
        # get the inputs
        inputs = X_train[i*batch_size:(i+1)*batch_size,:]
        labels = Y_train[i*batch_size:(i+1)*batch_size]
        # zero the parameter gradients
        mynet.zero_grad()
        # forward + backward + optimize
        outputs = mynet(inputs)
        loss = criterion(outputs, labels)
        grad = criterion.grad(outputs, labels)
        mynet.backward(grad)  
        mynet.step(optimizer)
        # print statistics
        running_loss += loss
        if i % 100 == 99: # print every 2000 mini-batches
            print('[{}, {}] - loss: {} | time: '.format(epoch+1, i+1, round(running_loss / 100, 3)),
                    round(timeit.default_timer() - start, 2))
            running_loss = 0.0
            start = timeit.default_timer()

print('Training dataset feature extraction')
Xout = X_train
N = Xout.shape[0]
X_features = np.empty((N, 400))
batch_size = 8
nb_batchs = int(N / batch_size)
for i in range(nb_batchs):
    inputs = Xout[i*batch_size:(i+1)*batch_size,:]
    outputs = mynet.features(inputs)
    X_features[i*batch_size:(i+1)*batch_size, :] = outputs.reshape(-1,16*5*5)
X_train = X_features.copy()

print('Test dataset feature extraction')
X_e = X_test
N = X_e.shape[0]
X_features = np.empty((N, 400))
batch_size = 8
nb_batchs = int(N / batch_size)
for i in range(nb_batchs):
    inputs = X_e[i*batch_size:(i+1)*batch_size,:]
    outputs = mynet.features(inputs)
    X_features[i*batch_size:(i+1)*batch_size, :] = outputs.reshape(-1,16*5*5)
X_test = X_features.copy()

#################################
###      CLASSIFICATION       ###
#################################
clf = svm.multiclass_ovo(C=1000., kernel=svm.Kernel.rbf(gamma=1/50), tol=1.0, max_iter=5000)

#################################
###        ///////////        ###
#################################
print('SVM training')
clf.fit(X_train, Y_train)

print('SVM prediction')
prediction = clf.predict(X_test)

prediction = pd.DataFrame(prediction)
prediction.reset_index(level=0, inplace=True)
prediction.columns = ['Id', 'Prediction']
prediction['Id'] = prediction['Id'] + 1
prediction['Prediction'] = prediction['Prediction'].astype(int)

prediction.to_csv('Yte.csv',sep=',', header=True, index=False)