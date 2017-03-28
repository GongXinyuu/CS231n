#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
#from cs231n.layer_utils_from3 import *

class ThreeLayerConvNet1(object):
# [conv-relu-pool(2x2)]x1 - conv -(bn)- relu - [affine]x2 - [softmax or SVM]
    def __init__(self, input_dim=(3, 32, 32), num_filters1=32, num_filters2=128, filter_size1=7, filter_size2=3,
             hidden_dims = 100, num_classes=10, weight_scale=1e-3, reg=0.0,
             dtype=np.float32):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        # self.bn_params = []  # list
        # self.bn_params = [{'mode': 'train'}]

        C, H, W = input_dim
        # conv
        self.params['W1'] = weight_scale * np.random.rand(num_filters1, C, filter_size1, filter_size1)  # F,C,H,W
        self.params['b1'] = np.zeros((1, num_filters1))  # (1,F1)
        # conv
        self.params['W2'] = weight_scale * np.random.rand(num_filters2, num_filters1, filter_size2, filter_size2)  # F,C,H,W
        self.params['b2'] = np.zeros((1, num_filters2))  # (1,F2)
        # affine
        self.params['W3'] = weight_scale * np.random.randn(num_filters2 * H * W / 4, hidden_dims)
        self.params['b3'] = np.zeros((1, hidden_dims))  # (1,F)
        # affine  在stride=1的情况下保证前后size不变（pad=(H-1)/2），除以4是因为前前一层通过了2 x 2的池化
        self.params['W4'] = weight_scale * np.random.randn(hidden_dims, num_classes)
        self.params['b4'] = np.zeros((1, num_classes))  # (1,F)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        # mode = 'test' if y is None else 'train'
        # for bn_param in self.bn_params:
        #     bn_param[mode] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        # gamma, beta = self.params['gamma'], self.params['beta']
        # pass conv_param to the forward pass for the convolutional layer
        filter_size1 = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}
        filter_size2 = W2.shape[2]
        conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}
        #这保证conv前后图像size不变

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # compute the forward pass
        l1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)
        l2, cache2 = conv_relu_forward(l1, W2, b2, conv_param2)
        l3, cache3 = affine_forward(l2, W3, b3)
        scores, cache4 = affine_forward(l3, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}
        # compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)
        dl3, dW4, db4 = affine_backward(dscores, cache4)
        dl2, dW3, db3 = affine_backward(dl3, cache3)
        dl1, dW2, db2 = conv_relu_backward(dl2, cache2)
        dX, dW1, db1 = conv_relu_pool_backward(dl1, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4}

        return loss, grads

class ThreeLayerConvNet2(object):
# [conv-relu-pool]X2 - [affine]X2 - [softmax or SVM]
    def __init__(self, input_dim=(3, 32, 32), num_filters1=64, num_filters2=64, filter_size1=7, filter_size2=3,
             hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
             dtype=np.float32):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim


        # conv
        self.params['W1'] = weight_scale * np.random.rand(num_filters1, C, filter_size1, filter_size1)  # F,C,H,W
        self.params['b1'] = np.zeros((1, num_filters1))  # (1,F1)
        # conv
        self.params['W2'] = weight_scale * np.random.rand(num_filters2, num_filters1, filter_size2,
                                                          filter_size2)  # F,C,H,W
        self.params['b2'] = np.zeros((1, num_filters2))  # (1,F2)
        # affine  在stride=1的情况下保证前后size不变（pad=(H-1)/2），除以16是因为之前通过了两个2 x 2的池化
        self.params['W3'] = weight_scale * np.random.randn(num_filters2 * H * W / 16, hidden_dim)
        self.params['b3'] = np.zeros((1, hidden_dim))  # (1,h)
        #affine
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b4'] = np.zeros((1, num_classes))  # (1,c)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):


        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size1 = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}
        filter_size2 = W2.shape[2]
        conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}
        # 这保证conv前后图像size不变

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # compute the forward pass
        l1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)
        l2, cache2 = conv_relu_pool_forward(l1, W2, b2, conv_param2, pool_param)
        a3, cache3 = affine_forward(l2, W3, b3)
        scores, cache4 = affine_forward(a3, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}
        # compute the backward pass
        #   foward层里的参数的导数即为backward所输出的参数
        data_loss, dscores = softmax_loss(scores, y)
        da3, dW4, db4 = affine_backward(dscores, cache4)
        dl2, dW3, db3 = affine_backward(da3, cache3)
        dl1, dW2, db2 = conv_relu_pool_backward(dl2, cache2)
        dX, dW1, db1 = conv_relu_pool_backward(dl1, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4}

        return loss, grads

class ThreeLayerConvNet3(object):
#[conv-relu-conv-relu-pool]x1 - [affine]x2 - [softmax or SVM]
    def __init__(self, input_dim=(3, 32, 32), num_filters1=64, num_filters2=64, filter_size1=7, filter_size2=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim
        # conv
        self.params['W1'] = weight_scale * np.random.rand(num_filters1, C, filter_size1, filter_size1)  # F,C,H,W
        self.params['b1'] = np.zeros((1, num_filters1))  # (1,F1)
        # conv
        self.params['W2'] = weight_scale * np.random.rand(num_filters2, num_filters1, filter_size2,
                                                          filter_size2)  # F,C,H,W
        self.params['b2'] = np.zeros((1, num_filters2))  # (1,F2)
        # affine  在stride=1的情况下保证前后size不变（pad=(H-1)/2），除以16是因为之前通过了两个2 x 2的池化
        self.params['W3'] = weight_scale * np.random.randn(num_filters2 * H * W / 4, hidden_dim)
        self.params['b3'] = np.zeros((1, hidden_dim))  # (1,h)
        # affine
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b4'] = np.zeros((1, num_classes))  # (1,c)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        # pass conv_param to the forward pass for the convolutional layer
        filter_size1 = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}
        filter_size2 = W2.shape[2]
        conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}
        # 这保证conv前后图像size不变

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # compute the forward pass
        l1, cache1 = conv_relu_forward(X, W1, b1, conv_param1)
        l2, cache2 = conv_relu_pool_forward(l1, W2, b2, conv_param2, pool_param)
        a3, cache3 = affine_forward(l2, W3, b3)
        scores, cache4 = affine_forward(a3, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}
        # compute the backward pass
        #   foward层里的参数的导数即为backward所输出的参数
        data_loss, dscores = softmax_loss(scores, y)
        da3, dW4, db4 = affine_backward(dscores, cache4)
        dl2, dW3, db3 = affine_backward(da3, cache3)
        dl1, dW2, db2 = conv_relu_pool_backward(dl2, cache2)
        dX, dW1, db1 = conv_relu_backward(dl1, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4}

        return loss, grads