#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW_each = np.zeros_like(W)  #shape (D, C)
  num_train, dim = X.shape
  num_class = W.shape[1]
  f = X.dot(W)  # N by C
  # Considering the Numeric Stability
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))  # N by 1
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)  # N by C
  y_trueClass = np.zeros_like(prob) # N x C
  y_trueClass[np.arange(num_train), y] = 1.0  #只将正确分类的置1的 N x C矩阵
  for i in xrange(num_train):
    for j in xrange(num_class):
      loss += -(y_trueClass[i, j] * np.log(prob[i, j])) #loss只包括正确分类概率的对数
      dW_each[:, j] = (prob[i, j] - y_trueClass[i, j]) * X[i, :]
    dW += dW_each
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  f = X.dot(W)  # N by C
  # Considering the Numeric Stability
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))  # N by 1
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) #计算各分类概率
  y_trueClass = np.zeros_like(prob)
  y_trueClass[range(num_train), y] = 1.0  # N by C
  loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W) #将所有正确分类的项计入loss，y_trueClass 相当于mask
  dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W  #根据推导的公式可写出
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

