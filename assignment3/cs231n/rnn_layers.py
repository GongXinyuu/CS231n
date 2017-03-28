#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  b_row = np.reshape(b, (1, b.shape[0]))
  next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b_row)
  cache = (x, Wx, Wh, prev_h, next_h) #!!!()
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache  #cache是tuple


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, Wx, Wh, prev_h, next_h = cache
  dtanh = (1 - next_h ** 2) * dnext_h  # (N, H)
  dx = np.dot(dtanh, Wx.T)  # (N, D)
  dprev_h = np.dot(dtanh, Wh.T)   # (N, H)
  dWx = np.dot(x.T, dtanh)   # (D, H)
  dWh = np.dot(prev_h.T, dtanh)   # (H, H)
  db = np.sum((dtanh), axis=0)  #(1, H)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  H = np.shape(b)[0]
  h = np.zeros((N, T, H))
  h_tem = h0
  cache = []  #list
  for i in xrange(T):
    h[:, i, :], cache_tem = rnn_step_forward(x[:, i, :], h_tem, Wx, Wh, b)
    h_tem = h[:, i, :]
    cache.append(cache_tem) #将返回的tupple作为list中的一个element
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache #cache.shape=(T, 5)


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  #这里的dh应该是从每个y回传的梯度
  x, Wx, Wh, prev_h, next_h = cache[-1] #最后一个
  _, D = x.shape
  N, T, H = dh.shape
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros(H)
  dprev_h_tem = np.zeros((N, H))  #内部传递的dprev_h，初始化为0
  for i in xrange(T): #xrange(start, stop, step),将会遍历从T -1 到 0
    dx_tem, dprev_h_tem, dWx_tem, dWh_tem, db_tem = rnn_step_backward(dh[:, T-i-1, :] + dprev_h_tem, cache.pop()) #pop将最后一层弹出
    dx[:, T-i-1, :] = dx_tem
    dWx += dWx_tem
    dWh += dWh_tem
    db += db_tem
  dh0 = dprev_h_tem
    #不断将残差累积回传，直至到达第一个
    #所有rnn单元其实是同一个(参数相同，以上是展开形式)，因此求梯度时需要将梯度不断累加，每一个单元的update均会叠加

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  # 每个words由integer表示，这里做的便是将其映射至vector
  N, T = x.shape
  V, D = W.shape
  out = np.zeros((N, T, D))
  for n in xrange(N):
    for t in xrange(T):
      out[n, t, :] = W[x[n, t]] #取W中的第x[n, t]行，作为其vector，其中max(x[n, t])<=V  (N, T, D)
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  N, T, D = dout.shape
  dW = np.zeros(W.shape)
  for n in xrange(N):
    for t in xrange(T):
      dW[x[n, t]] += dout[n, t, :]  #与forward类似
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N, H = np.shape(prev_h)
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b # (N, 4H)
  a_i, a_f, a_o, a_g = a[:, 0:H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]  #(N, H) 从标记的头开始，到结尾之前，因此不减1
  i = sigmoid(a_i)  #(N, H)
  f = sigmoid(a_f)  #(N, H)
  o = sigmoid(a_o)  #(N, H)
  g = np.tanh(a_g)  #(N, H)
  next_c = f * prev_c + i * g  #(N, H)
  next_h = o * np.tanh(next_c)  #(N, H)
  cache = (x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, next_h)  #tuple
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  # x: Input data, of shape (N, D)
  x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, next_h = cache
  N, H = np.shape(dnext_h)
  _, D = np.shape(x)
  dnext_c += dnext_h * o * (1 - (np.tanh(next_c)) ** 2) #后面一项是从next_h流回的分量
  # i
  di = dnext_c * g  # (N, H)
  da_i = i * (1 - i) * di # (N, H)
  dWx_i = np.dot(x.T, da_i)  # (D, H)
  dWh_i = np.dot(da_i.T, prev_h).T  # (H, H)  #注意H x H的顺序
  db_i = np.sum(da_i, axis=0)
  # f
  df = dnext_c * prev_c  # (N, H)
  da_f = f * (1 - f) * df
  dWx_f = np.dot(x.T, da_f)  # (D, H)
  dWh_f = np.dot(da_f.T, prev_h).T  # (H, H)
  db_f = np.sum(da_f, axis=0)
  # o
  do = dnext_h * np.tanh(next_c)  # (N, H)
  da_o = o * (1 - o) * do  # (N, H)
  dWx_o = np.dot(x.T, da_o)  # (D, H)
  dWh_o = np.dot(da_o.T, prev_h).T  # (H, H)
  db_o = np.sum(da_o, axis=0)  # H
  # g
  dg = dnext_c * i
  da_g = (1 - g**2) * dg
  dWx_g = np.dot(x.T, da_g)  # (D, H)
  dWh_g = np.dot(da_g.T, prev_h).T  # (H, H)
  db_g = np.sum(da_g, axis=0)

  dWx = np.zeros((D, 4 * H))
  dWh = np.zeros((H, 4 * H))
  db = np.zeros(4 * H)
  dWx[:, 0:H], dWx[:, H:2 * H], dWx[:, 2 * H:3 * H], dWx[:, 3 * H:] = dWx_i, dWx_f, dWx_o, dWx_g
  dWh[:, 0:H], dWh[:, H:2 * H], dWh[:, 2 * H:3 * H], dWh[:, 3 * H:] = dWh_i, dWh_f, dWh_o, dWh_g
  db[0:H], db[H:2 * H], db[2 * H:3 * H], db[3 * H:] = db_i, db_f, db_o, db_g

  da = np.hstack((da_i, da_f, da_o, da_g))  # (N, 4H)
  dx = da.dot(Wx.T)  # (N, D)
  dprev_h = da.dot(Wh.T)
  dprev_c = dnext_c * f # (N, H)

  # _, H = dnext_h.shape
  # x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, next_h = cache
  # ai, af, ao, ag = a[:, 0:H], a[:, H:2 * H], a[:, 2 * H:3 * H], a[:, 3 * H:]
  # dnext_c += dnext_h * o * (1 - (np.tanh(next_c)) ** 2)
  # do = dnext_h * np.tanh(next_c)
  # df = dnext_c * prev_c
  # dprev_c = dnext_c * f
  # di = dnext_c * g
  # dg = dnext_c * i
  # dai = di * (i * (1 - i))
  # daf = df * (f * (1 - f))
  # dao = do * (o * (1 - o))
  # dag = dg * (1 - g ** 2)
  # da = np.hstack((dai, daf, dao, dag))  # (N, 4H)
  # dx = da.dot(Wx.T)  # (N, D)
  # dWx = x.T.dot(da)
  # dprev_h = da.dot(Wh.T)
  # dWh = prev_h.T.dot(da)
  # db = np.sum(da, axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = np.shape(x)
  _, H = np.shape(h0)
  prev_h = h0
  prev_c = 0  #t he initial cell state is passed as input, but the initial cell state is set to zero
  cache = []  #列表
  h = np.zeros((N, T, H))
  #next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
  for i in xrange(T):
    prev_h, prev_c, cache_tem = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
    h[:,i,:] = prev_h
    cache.append(cache_tem) #向列表中添加tuple

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  # dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache)
  # 这里的dh是由外部输入的分量
  x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, next_h = cache[0]
  N, T, H = np.shape(dh)
  N, D = x.shape
  dx = np.zeros((N, T, D))
  dWx = np.zeros((D, 4 * H))
  dWh = np.zeros((H, 4 * H))
  db = np.zeros(4 * H)
  dprev_c = np.zeros((N, H))  #初始化为0?--同下
  dprev_h = np.zeros((N, H))  #内部传递的dprev_h，初始化为0?--因为梯度是由dloss流入的，该内部h并未与loss相接
  for i in xrange(T):
    dx_tem, dprev_h, dprev_c, dWx_tem, dWh_tem, db_tem = lstm_step_backward(dh[:, T - i - 1,:] + dprev_h, dprev_c, cache.pop())
    dx[:, T-i-1, :] = dx_tem
    dWx += dWx_tem
    dWh += dWh_tem
    db += db_tem
  dh0 = dprev_h #传到第一个即为dh0
  #所有LSTM单元其实是同一个(参数相同，以上是展开形式)，因此求梯度时需要将梯度不断累加，每一个单元的update均会叠加
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

