#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:42:31 2017

@author: EmileMathieu
"""

import numpy as np
import math

class Optimizer(object):
    def __init__(self, lambda_reg, batch):
        self._reset_state()
        self.lambda_reg = lambda_reg
        self.batch = batch
    def __call__(self, layer_id, weight_type, objective, grad):
        raise NotImplementedError()
    def zero_grad(self):
        self._reset_state()
    def _reset_state(self):
        self.state = {}

class SGD(Optimizer):
    def __init__(self, lr=0.1, momentum=0, lambda_reg=0, batch=16):
        super().__init__(lambda_reg, batch)
        self.lr = lr
        self.momentum = momentum
        
    def get_state(self, key):
        if self.momentum != 0 and key in self.state:
            mu = self.state[key]
        else:
            mu = 0
        return mu

    def __call__(self, layer_id, weight_type, objective, grad):
        key = str(layer_id) + weight_type
        old_v = self.get_state(key)
        new_v = self.lr * grad

        new_v += self.momentum * old_v
        self.state[key] = new_v
        return objective - new_v

class Adam(Optimizer):
    """
    ADAM Gradient descent optimization algorithm
    -------------
    Parameters:
        lr  (float, default: 0.001): 
            learning rate of the gradient descent
        betas   (tuple[float, float], default: [0.9, 0.999]):
            decay rates for first and second order momentums
        eps     (float, default: 10e-8): 
            term for numerical stability

    """
    def __init__(self, lr=0.001, betas=[0.9, 0.999], eps=10e-8, lambda_reg=0, batch=16):
        super().__init__(lambda_reg, batch)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
    def get_state(self, key, shape):
        if key in self.state:
            m, v, step = self.state[key]
        else:
            m = np.zeros(shape)
            v = np.zeros(shape)
            step = 0
        return m, v, step

    def __call__(self, layer_id, weight_type, objective, grad):
        """
        Performs a step of Gradient descent with ADAM optimization
        -------------
        Pameters:
            gradient (array)
              Gradient of the function to optimize
            objective (array):
              Initial point
        """
        key = str(layer_id) + weight_type
        m, v, step = self.get_state(key, grad.shape)
        step = step + 1
        beta1 = self.betas[0]
        beta2 = self.betas[1]
        m = beta1 * m + (1 - beta1) * grad ## Estimate of the mean of the gradient
        v = beta2 * v + (1 - beta2) * grad * grad ## Estimate of the variance of the gradient
        self.state[key] = m, v, step
        ## Correct biases
        m_unbiased = m / (1 - beta1)
        v_unbiased = v / (1 - beta2)
        ## Compute step size
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1
        ## Define new step
        objective = (1 - self.lambda_reg*step_size/self.batch) * objective - step_size / (np.sqrt(v_unbiased) + self.eps) * m_unbiased
        return objective

class RMSprop(Optimizer):
    def __init__(self, lr=0.01, alpha=0.99, eps=10e-8, lambda_reg=0, batch=16):
        super().__init__(lambda_reg, batch)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def get_state(self, key, shape):
        if key in self.state:
            square_avg, step = self.state[key]
        else:
            square_avg = np.zeros(shape)
            step = 0
        return square_avg, step

    def __call__(self, layer_id, weight_type, objective, grad):
        key = str(layer_id) + weight_type
        square_avg, step = self.get_state(key, grad.shape)
        step = step + 1
        square_avg = self.alpha * square_avg + (1 - self.alpha) * grad * grad
        self.state[key] = square_avg, step
        avg = np.sqrt(square_avg + self.eps)

        return (1 - self.lambda_reg*self.lr/self.batch) * objective - self.lr * grad / avg