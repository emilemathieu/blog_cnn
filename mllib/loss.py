import numpy as np

class CrossEntropyLoss(object):
    def __call__(self, Y, labels):
        loss = 0
        for i, y in enumerate(Y):
            loss += - y[labels[i]] + np.log(np.sum(np.exp(y)))
        return loss/len(labels)
    
    def grad(self, Y, labels):
        output_grad = np.empty_like(Y)
        for i, y in enumerate(Y):
            output_grad[i,:] = np.exp(y) / np.sum(np.exp(y))
            output_grad[i, labels[i]] -= 1
        return output_grad