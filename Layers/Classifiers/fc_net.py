import numpy as np

from Layers.layers import *
from Layers.layer_utils import *


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
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
 
    self.params['W1'] = weight_scale*np.random.randn(input_dim,hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b2'] = np.zeros(num_classes)


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
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    out, cache_a1 = affine_forward(X,W1,b1)
    out, cache_relu =  relu_forward(out)
    scores, cache_a2 = affine_forward(out,W2,b2)

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    data_loss, dx = softmax_loss(scores,y)
    dx, dW2, db2 = affine_backward(dx,cache_a2)
    dx = relu_backward(dx,cache_relu)
    dx, dW1, db1 = affine_backward(dx,cache_a1)

    dW1 += self.reg*W1
    dW2 += self.reg*W2
    reg_loss = 0.5*self.reg*sum(np.sum(W*W) for W in[W1,W2])
    loss = data_loss + reg_loss
    grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2}
    return loss, grads

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(x, w, b, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    arelu, relu_cache = relu_forward(a)
    out, do_cache = dropout_forward(arelu, dropout_param)
    cache = (fc_cache, relu_cache, do_cache)
    return out, cache


def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache,  relu_cache, do_cache= cache
    a_do = dropout_backward(dout, do_cache)
    da = relu_backward(a_do, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
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
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    in_dim = input_dim
    for i in range(1,self.num_layers):
    	out_dim = hidden_dims[i-1]
        self.params["W%d" % i] = weight_scale*np.random.randn(in_dim,out_dim)
        self.params["b%d" % i] =np.zeros(out_dim)
        in_dim = out_dim
    i+=1
    self.params["W%d" % i] = weight_scale*np.random.randn(in_dim,num_classes)
    self.params["b%d" % i] = np.zeros(num_classes)
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
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
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    inp = X
    caches = []
    for i in range(1, self.num_layers):
    	W, b = self.params["W%d" % i], self.params["b%d" % i]
        if self.use_dropout:
        	out, cache = affine_relu_dropout_forward(inp, W, b,self.dropout_param)	
	else:	
		out, cache = affine_relu_forward(inp,W,b)
        inp = out
        caches.append(cache)
    i += 1
    W, b = self.params["W%d" % i], self.params["b%d" % i]
    scores, cache = affine_forward(inp, W, b)
    caches.append((cache, None)

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss, dout = softmax_loss(scores, y)
    for i in range(1, self.num_layers + 1):
    	loss += 0.5 * self.reg * (self.params["W%d" % i] ** 2).sum()

    W = self.params["W%d" % self.num_layers]
    grads["W%d" % self.num_layers] = caches[-1][0][0].T.dot(dout) + self.reg * W
    grads["b%d" % self.num_layers] = dout.sum(axis=0)
    delta = dout.dot(W.T)
    for i in range(self.num_layers - 1, 0, -1):
	cache = caches[i-1]
	if self.use_dropout:
		dx, dW, db = affine_relu_dropout_backward(delta, cache)
	else:
		dx, dW, db = affine_relu_backward(delta, cache)
	W = self.params["W%d" % i]
	grads["W%d" % i] = dW + self.reg * W
	grads["b%d" % i] = db
	delta = dx

    return loss, grads
