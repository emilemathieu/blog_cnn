#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:29:39 2017

@author: EmileMathieu
"""
import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from .im2col_cyt import col2im_cython, im2col_cython

class Module(object):
    """ Base class for neural network's layers
    """
    def forward(self, X):
        """ Apply the layer function to the input data
        Parameters
        ----------
        X : array-like, shape = [n_samples, depth_in, height_in, width_in]
        Returns
        -------
        transformed data : array-like, shape = [n_samples, depth_out, height_out, width_out]
        """
        raise NotImplementedError()

    def __call__(self, X):
        return self.forward(X)

    def backward(self, output_grad):
        """ Compute the gradient of the loss with respect to its parameters and to its input
        Parameters
        ----------
        output_grad : array-like, shape = [n_samples, depth_out, height_out, width_out]
                      gradient returned by the above layer.
        Returns
        -------
        gradient : array-like, shape = [n_samples, depth_in, height_in, width_in]
                   gradient to be forwarded to bottom layers
        """
        raise NotImplementedError()

    def step(self, optimizer):
        """ Do an optimization step in the direction given by the optimizer
        Parameters
        ----------
        optimizer : instance of Optimizer
        """
        self._bias = optimizer(id(self), 'bias', self._bias, self._grad_bias)
        self._weight = optimizer(id(self), 'weight', self._weight, self._grad_weight)

    def zero_grad(self):
        """ Reset gradient of the layer's parameters
        """
        self._grad_bias = None
        self._grad_weight = None

class ReLU(Module):
    """ Applies the rectified linear unit function element-wise ReLu(x) = max(0,x)
    Non-linear Activation layer
    """
    def func(self, X):
         return np.maximum(X, 0)

    def forward(self, X):
        self._last_input = X
        return self.func(X)

    def backward(self, output_grad):
        return output_grad * self.func(self._last_input)

class MaxPool2d(Module):
    """ Applies a 2D max pooling over an input signal composed of several input planes.
    Im2Col approach http://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
    Parameters
    ----------
    kernel_size : int
        The size of the window to take a max over
    stride : int
        The stride of the window
    """
    def __init__(self, kernel_size, stride=2, padding=0):
         self._kernel_size = kernel_size
         self._padding = padding
         self._stride = stride

    def forward(self, X):
         N,d,h,w = X.shape
         self._X_shape = N,d,h,w
         h_out = int((h - self._kernel_size)/self._stride + 1)
         w_out = int((w - self._kernel_size)/self._stride + 1)

         X_reshaped = X.reshape(N*d, 1, h, w)
         X_col = im2col_cython(X_reshaped, self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
         max_idx = np.argmax(X_col, axis=0)
         self._max_idx = max_idx
         res = X_col[max_idx, range(max_idx.size)]
         res = res.reshape(h_out, w_out, N, d)
         res = res.transpose(2, 3, 0, 1)
         return res

    def backward(self, output_grad):
        N,d,h,w = self._X_shape
        kernel_area = self._kernel_size*self._kernel_size
        dX_col = np.zeros((kernel_area, int(N*d*h*w/kernel_area)))
        dout_flat = output_grad.transpose(2, 3, 0, 1).ravel()
        dX_col[self._max_idx, range(self._max_idx.size)] = dout_flat
        dX = col2im_cython(dX_col, N * d, 1, h, w, self._kernel_size, self._kernel_size, padding=0, stride=self._stride)
        return dX.reshape(self._X_shape)

class Conv2d(Module):
    """ Applies a 2D convolution over an input signal composed of several input planes.
    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : int
        Size of the convolving kernel
    stride : int, optional (default=1)
        Stride of the convolution
    padding : int, optional (default=0)
        Zero-padding added to both sides of the input
    Variables
    ----------
    weight : array-like, shape = [out_channels, in_channels, kernel_size, kernel_size]
        the learnable weights of the module
    bias : array-like, shape = [out_channels]
        the learnable bias of the module
    """
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
         self._in_channels = in_channels
         self._out_channels = out_channels
         self._kernel_size = kernel_size
         self._stride = stride
         self._padding = padding
         law_bound = 1/np.sqrt(kernel_size*kernel_size*in_channels)
         self._bias = np.random.uniform(-law_bound,law_bound,size=(out_channels)).astype(np.float32)
         self._weight = np.random.uniform(-law_bound,law_bound,size=(out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)

    def forward(self, X):
        N,d,h,w = X.shape
        self._X_shape = N,d,h,w
        n_filters = self._weight.shape[0]
        h_out = int((h - self._kernel_size + 2*self._padding) / self._stride + 1)
        w_out = int((w - self._kernel_size + 2*self._padding) / self._stride + 1)

        # Im2Col approach: http://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
        # (https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html)
        ###X_col = im2col_indices(X, self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
        X_col = im2col_cython(X, self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
        self._X_col = X_col
        W_col = self._weight.reshape(n_filters, -1)
        res = W_col @ X_col
        res = res + np.tile(self._bias, (res.shape[1],1)).T
        res = res.reshape(n_filters, h_out, w_out, N)
        res = res.transpose(3, 0, 1, 2)
        return res

    def backward(self, output_grad):
        n_filters = self._weight.shape[0]

        self._grad_bias = np.sum(output_grad, axis=(0, 2, 3))

        output_grad_reshaped = output_grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        grad_weight = output_grad_reshaped @ self._X_col.T
        self._grad_weight = grad_weight.reshape(self._weight.shape)

        W_reshape = self._weight.reshape(n_filters, -1)
        dX_col = W_reshape.T @ output_grad_reshaped
        ###dX = col2im_indices(dX_col, self._X_shape, self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
        dX = col2im_cython(dX_col, self._X_shape[0], self._X_shape[1], self._X_shape[2], self._X_shape[3], self._kernel_size, self._kernel_size, padding=self._padding, stride=self._stride)
        return dX

class Linear(Module):
    """ Applies a linear transformation to the incoming data: y=Ax+b
    Parameters
    ----------
    in_features : int
        size of each input sample
    out_features : int
        size of each output sample
    Variables
    ----------
    weight : array-like, shape = [out_features x in_features]
        the learnable weights of the module
    bias : array-like, shape = [out_features]
        the learnable bias of the module
    """
    def __init__(self,in_features, out_features):
        self._in_features = in_features
        self._out_features = out_features
        law_bound = 1/np.sqrt(in_features)
        self._bias = np.random.uniform(-law_bound,law_bound,size=(out_features)).astype(np.float32)
        self._weight = np.random.uniform(-law_bound,law_bound,size=(out_features, in_features)).astype(np.float32)

    def forward(self, X):
        self._last_input = X
        return np.matmul(X, self._weight.T) + np.tile(self._bias, (X.shape[0],1))

    def backward(self, output_grad):
        self._grad_bias = np.sum(output_grad,axis=0)
        self._grad_weight = np.dot(output_grad.T, self._last_input)
        return np.dot(output_grad, self._weight)
    
class BatchNorm2d(Module):
    """ Applies Batch Normalization over a 4d input that is seen as a mini-batch of 3d inputs
    Useful to speed up the learning
    See http://cthorey.github.io./backpropagation/
    or https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    Parameters
    ----------
    num_features : int
        num_features from an expected input of size batch_size x num_features x height x width
    eps : float, optional (default=0.04)
        a value added to the denominator for numerical stability
    momentum : float, optional (default=0.1)
        the value used for the running_mean and running_var computation. 
    Variables
    ----------
    weight : array-like, shape = [num_features]
        the learnable weights of the module
    bias : array-like, shape = [num_features]
        the learnable bias of the module

    """
    def __init__(self, in_features, eps=0.04, momentum=0.1):
        self.in_features = in_features
        self._weight = np.random.uniform(size=in_features).astype(np.float32)
        self._bias = np.zeros(in_features).astype(np.float32)
        self.eps = eps #10/255
        self.momentum = momentum
        self._mu = None
        self._sigma2 = None
    def forward(self, X):
        self._last_input = X
        N, D, H, W = X.shape
        mu = 1/N*np.sum(X,axis =0) # Size (H,)
        if self._mu is not None:
            mu = (1 - self.momentum) * mu + self.momentum * self._mu
        self._mu = mu
        sigma2 = 1/N*np.sum((X-mu)**2,axis=0)# Size (H,) 
        if self._sigma2 is not None:
            sigma2 = (1 - self.momentum) * sigma2 + self.momentum * self._sigma2
        self._sigma2 = sigma2
        hath = (X-mu)*(sigma2+self.eps)**(-1./2.)
        res = np.tile(self._weight, (H,W,1)).swapaxes(0,2) * hath
        return res + np.tile(self._bias, (H,W,1)).swapaxes(0,2)

    def backward(self, output_grad):
        N, D, H, W = output_grad.shape
        weight_big = np.tile(self._weight, (H,W,1)).swapaxes(0,2)
        dy = output_grad
        h = self._last_input
        mu = self._mu
        var = self._sigma2
        self._grad_bias = np.sum(dy, axis=(0,2,3))
        self._grad_weight = np.sum((h - mu) * (var + self.eps)**(-1. / 2.) * dy, axis=(0,2,3))
        return (1. / N) * weight_big * (var + self.eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
            - (h - mu) * (var + self.eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
    
class Flatten(Module):
    """ Flatten a multi-dimensional tensor to a 1 dimensional vector
    """
    def forward(self, X):
        self._X_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, output_grad):
        return output_grad.reshape(self._X_shape)

class Sequential(Module):
    """ Special instance of neural network which can be constructed as a sequence of layers
    """
    def __init__(self, *modules):
        self._modules = list(modules)

    def add_module(self, module):
        self._modules.append(module)

    def forward(self, X):
        for module in self._modules:
            X = module.forward(X)
        return X

    def backward(self, output_grad):
        for module in reversed(self._modules):
            output_grad = module.backward(output_grad)
        return output_grad

    def has_parameters(self, module):
        if isinstance(module, Linear) or isinstance(module, Conv2d) or isinstance(module, BatchNorm2d):
            return True
        else:
            return False

    def step(self, optimizer):
        for module in self._modules:
            if self.has_parameters(module):
                module.step(optimizer)

    def zero_grad(self):
        for module in self._modules:
            if self.has_parameters(module):
                module.zero_grad()
                
    def zero_momentum(self):
        for module in self._modules:
            if isinstance(module, BatchNorm2d):
                module.zero_momentum()

    def parameters(self):
        parameters = []
        for module in self._modules:
             if self.has_parameters(module):
                parameters.append(module._weight)
                parameters.append(module._bias)
        return parameters