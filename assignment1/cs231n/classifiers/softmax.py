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
  num_data = np.shape(X)[0]
  num_class = np.shape(W)[1]
  num_dim = np.shape(X)[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_data):
    scores = W.T @ X[i,:]   # scores: (C,)
    scores -= np.max(scores)
    loss += np.log(np.sum(np.exp(scores))) - scores[y[i]]
    for j in range(num_class):
        if j != y[i]:
            dW[:, j] += np.exp(scores[j]) * X[i,:] / np.sum(np.exp(scores))
    dW[:, y[i]] += -X[i,:] + np.exp(scores[y[i]]) * X[i,:] / np.sum(np.exp(scores))
    
  loss /= num_data
  dW /= num_data
  loss += reg * np.sum(W * W)  
  dW += 2 * reg * W
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
  dW = np.zeros_like(W) # dW: (D, C)
  num_data = np.shape(X)[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X @ W  # scores: (N, C)
  scores -= np.mean(scores, axis=1).reshape(-1, 1)
  loss = np.log(np.sum(np.exp(scores), axis=1)).sum() - \
         np.choose(y, scores.T).sum()
#   index_data = np.argsort(y) #index_data: (N,)
#   y_sorted = y[index_data]
#   vals, idx_start, count = unique(y_sorted, return_counts=True,
#                                 return_index=True)
#   index_data_repeat = np.split(index_data, idx_start[1:])
#   X_sorted_split= np.split(X.T[:,index_data], indx_start[1:], axis=1)
#   X_sorted_split_sum = np.concatenate(np.sum(X_sorted_split, axis=1))
  dW_add = np.zeros(np.shape(scores))
  dW_add[np.arange(num_data), y] = 1   # size:(N, C)
  dW = X.T @ (np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)) - X.T @ dW_add
    
  loss /= num_data
  dW /= num_data
  loss += reg * np.sum(W * W)  
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

