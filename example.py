import sys, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mllib import tools, nn, optim, loss, mnist

#################################
###          LOAD DATA        ###
#################################
print('Load dataset...')

mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

W = int(np.sqrt(X_train.shape[1]))
X_train = X_train.reshape(-1, 1, W , W)
X_train = X_train / 256.
X_test = X_test.reshape(-1, 1, W , W)
X_test = X_test / 256.

#################################
###      MODEL DEFINITION     ###
#################################
print('Model definition...')

class MyNet(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 120),
            nn.ReLU(),
            nn.Linear(120, 10),
            nn.ReLU()
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

#################################
###       MODEL TRAINING      ###
#################################
print('Model training...')

mynet = MyNet()
optimizer = optim.RMSprop(lr=0.001)
objective = loss.CrossEntropyLoss()

batch_size = 64
nb_iterations = 1

def train(X_train, Y_train, batch_size):
    N_train = X_train.shape[0]
    nb_batchs_train = int(N_train / batch_size)

    running_loss = 0.0
    start = timeit.default_timer()
    suffle = np.random.permutation(N_train)
    X_train = X_train[suffle,:]
    Y_train = Y_train[suffle]
    for i in range(nb_batchs_train):
        # get the inputs
        inputs = X_train[i*batch_size:(i+1)*batch_size,:]
        labels = Y_train[i*batch_size:(i+1)*batch_size]
        # zero the parameter gradients
        # forward + backward + optimize
        outputs = mynet(inputs)
        loss = objective(outputs, labels)
        grad = objective.grad(outputs, labels)
        mynet.backward(grad)  
        mynet.step(optimizer)
        # print statistics
        running_loss += loss
        if i % 100 == 99: # print every 2000 mini-batches
            print('[{}, {}] - loss: {} | time: '.format(epoch+1, i+1, round(running_loss / 100, 3)),
                    round(timeit.default_timer() - start, 2))
            running_loss = 0.0
            start = timeit.default_timer()

def test(X_test, Y_test, batch_size):
    N_test = X_test.shape[0]
    nb_batchs_test = int(N_test / batch_size)

    running_loss = 0.0
    start = timeit.default_timer()
    suffle = np.random.permutation(N_test)
    X_test = X_test[suffle,:]
    Y_test = Y_test[suffle]
    for i in range(nb_batchs_test):
        # get the inputs
        inputs = X_test[i*batch_size:(i+1)*batch_size,:]
        labels = Y_test[i*batch_size:(i+1)*batch_size]
        # forward + backward + optimize
        outputs = mynet(inputs)
        loss = objective(outputs, labels)
        # print statistics
        running_loss += loss
    print('[{}, TEST] - loss: {} | time: '.format(epoch+1, round(running_loss / 100, 3)),
            round(timeit.default_timer() - start, 2))    

start_global = timeit.default_timer()
for epoch in range(0, nb_iterations, 1): # loop over the dataset multiple times
    test(X_test, Y_test, batch_size)
    train(X_train, Y_train, batch_size)

#################################
###            PLOTS          ###
#################################

outputs = mynet(X_test[[0], :, :, :])
probs = np.exp(outputs) / np.sum(np.exp(outputs))

pixels = np.array(X_test[0, 0, :, :] * 256, dtype='uint8')
plt.imshow(pixels, cmap='gray')
plt.savefig('digit.png')

plt.clf()
plt.bar(range(10), probs[0])
plt.savefig('pred.png')