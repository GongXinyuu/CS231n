#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]#C
  num_train = X.shape[0]#N
  loss = 0.0
  for i in xrange(num_train):#N
    scores = X[i].dot(W)#1xC
    correct_class_score = scores[y[i]]#C
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i, :]  # 根据公式：∇Wyi Li = - xiT(∑j≠yi1(xiWj - xiWyi +1>0)) + 2λWyi; compute the correct_class gradients
        dW[:, j] += X[i, :]  # 根据公式： ∇Wj Li = xiT 1(xiWj - xiWyi +1>0) + 2λWj , (j≠yi)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)#NxC
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores_correct = scores[np.arange(num_train), y]# 将label标记为正确的得分存入
  scores_correct = np.reshape(scores_correct, (num_train, 1))  # 转成Nx1矩阵
  margins = scores - scores_correct + 1.0#将scores矩阵与correct按列相减并加一  N x C
  margins[np.arange(num_train), y] = 0.0#将正确得分的margin项置零
  margins[margins <= 0] = 0.0#进行max(0,margins)运算
  loss=loss + np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)#正则项是L2
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins > 0] = 1.0#进行indicator1函数操作 N x C
  row_sum = np.sum(margins, axis=1)  # 1 by N；按列相加
  margins[np.arange(num_train), y] = -row_sum #将correct对应的位置根据公式进行sum计算
  dW += np.dot(X.T, margins) / num_train + reg * W  # D by C; 微分运算的第二步，乘上x，并除以num，加上正则项
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
