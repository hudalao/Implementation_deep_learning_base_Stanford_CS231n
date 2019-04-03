from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params = {}
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        N = np.shape(X)[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        out1, cache1 = affine_forward(X, W1, b1)
        out2, cache2 = relu_forward(out1)
        out3, cache3 = affine_forward(out2, W2, b2)
        scores = out3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout3 = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1) 
        loss += 0.5 * self.reg * np.sum(W2 * W2)
        
        dout2, dW2, db2 = affine_backward(dout3*N, cache3)
        dout1 = relu_backward(dout2, cache2)
        dx, dW1, db1 = affine_backward(dout1, cache1)
        
        grads['W2'] = dW2/N + self.reg * W2
        grads['b2'] = db2/N
        grads['W1'] = dW1/N + self.reg * W1
        grads['b1'] = db1/N
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    
def bn_relu_forward(x, gamma, beta, bn_param):
    an, bn_cache = batchnorm_forward(x, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (bn_cache, relu_cache)
    return out, cache
    
    
def bn_relu_backward(dout, cache):
    bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward(dan, bn_cache)
    return dx, dgamma, dbeta
    

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.num_hiddens = len(hidden_dims)

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        if self.normalization=='batchnorm':
            self.params['gamma1'] = np.ones(hidden_dims[0])
            self.params['beta1'] = np.zeros(hidden_dims[0])
        ind_hidlay = -1
        for ind_hidlay in np.arange(self.num_hiddens-1):
            self.params['W'+str(ind_hidlay+2)] = np.random.normal(scale=weight_scale, 
                                                                size=(hidden_dims[ind_hidlay], 
                                                                      hidden_dims[ind_hidlay+1]))
            self.params['b'+str(ind_hidlay+2)] = np.zeros(hidden_dims[ind_hidlay+1])
            if self.normalization=='batchnorm':
                self.params['gamma'+str(ind_hidlay+2)] = np.ones(hidden_dims[ind_hidlay+1])
                self.params['beta'+str(ind_hidlay+2)] = np.zeros(hidden_dims[ind_hidlay+1])
        self.params['W'+str(ind_hidlay+3)] = np.random.normal(scale=weight_scale, 
                                                              size=(hidden_dims[ind_hidlay+1], 
                                                                    num_classes))
        self.params['b'+str(ind_hidlay+3)] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        #if self.use_dropout:
        #    self.dropout_param = {'mode': 'train', 'p': dropout}
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

            
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        num_layers = self.num_layers
        num_hiddens = self.num_hiddens
        W = [0.] * num_layers
        b = [0.] * num_layers
        for ind in np.arange(num_layers):
            W[ind], b[ind] = self.params['W'+str(ind+1)], self.params['b'+str(ind+1)]
        if self.normalization=='batchnorm':
            gamma = [0.] * num_hiddens
            beta = [0.] * num_hiddens
            for ind in np.arange(num_hiddens):
                gamma[ind], beta[ind] = self.params['gamma'+str(ind+1)], self.params['beta'+str(ind+1)]
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        out = [0.] * (num_layers * 2 - 1)
        cache = [0.] * (num_layers * 2 - 1)
        cache_drop = [0.] * num_hiddens
        out[0], cache[0] = affine_forward(X, W[0], b[0])
        if self.normalization=='batchnorm':
            out[1], cache[1] = bn_relu_forward(out[0], gamma[0], beta[0], self.bn_params[0])
            out[1], cache_drop[0] = dropout_forward(out[1], self.dropout_param)
            for ind in np.arange(1, num_hiddens):
                out[ind*2], cache[ind*2] = affine_forward(out[ind*2-1], W[ind], b[ind])
                out[ind*2+1], cache[ind*2+1] = bn_relu_forward(out[ind*2], gamma[ind], beta[ind], self.bn_params[ind])
                out[ind*2+1], cache_drop[ind] = dropout_forward(out[ind*2+1], self.dropout_param)
        else:
            out[1], cache[1] = relu_forward(out[0])
            out[1], cache_drop[0] = dropout_forward(out[1], self.dropout_param)
            ind = 0
            for ind in np.arange(1, num_hiddens):
                out[ind*2], cache[ind*2] = affine_forward(out[ind*2-1], W[ind], b[ind])
                out[ind*2+1], cache[ind*2+1] = relu_forward(out[ind*2])
                out[ind*2+1], cache_drop[ind] = dropout_forward(out[ind*2+1], self.dropout_param)
        out[-1], cache[-1] = affine_forward(out[-2], W[ind+1], b[ind+1])
        scores = out[-1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        N = np.shape(X)[0]
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        dout = [0.] * num_layers * 2  # the dout[0] is dx
        dW = [0.] * num_layers
        db = [0.] * num_layers 
        if self.normalization=='batchnorm':
            dgamma = [0.] * num_hiddens
            dbeta = [0.] * num_hiddens
        
        loss, dout[-1] = softmax_loss(scores, y)
        dout[-2], dW[-1], db[-1] = affine_backward(dout[-1]*N, cache[-1])
        loss += 0.5 * self.reg * np.sum(W[-1] * W[-1]) #update the loss
        grads['W'+str(num_layers)] = dW[-1]/N + self.reg * W[-1] #update grads
        grads['b'+str(num_layers)] = db[-1]/N
        if self.normalization=='batchnorm':
            for ind in np.arange(num_hiddens-1, -1, -1):   #iterate over all the hidden layers
                dout[ind*2+2] = dropout_backward(dout[ind*2+2], cache_drop[ind])
                dout[ind*2+1], dgamma[ind], dbeta[ind] = bn_relu_backward(dout[ind*2+2], cache[ind*2+1])
                dout[ind*2], dW[ind], db[ind] = affine_backward(dout[ind*2+1], cache[ind*2])
                loss += 0.5 * self.reg * np.sum(W[ind] * W[ind])
                grads['W'+str(ind+1)] = dW[ind]/N + self.reg * W[ind] #update grads
                grads['b'+str(ind+1)] = db[ind]/N
                grads['gamma'+str(ind+1)] = dgamma[ind]/N
                grads['beta'+str(ind+1)] = dbeta[ind]/N
        else:
            for ind in np.arange(num_hiddens-1, -1, -1):   #iterate over all the hidden layers
                dout[ind*2+2] = dropout_backward(dout[ind*2+2], cache_drop[ind])
                dout[ind*2+1] = relu_backward(dout[ind*2+2], cache[ind*2+1])
                dout[ind*2], dW[ind], db[ind] = affine_backward(dout[ind*2+1], cache[ind*2])
                loss += 0.5 * self.reg * np.sum(W[ind] * W[ind])
                grads['W'+str(ind+1)] = dW[ind]/N + self.reg * W[ind] #update grads
                grads['b'+str(ind+1)] = db[ind]/N
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
