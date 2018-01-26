import numpy as np

from Layers.layers import *
from Layers.fast_layers import *
from Layers.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
   
    self.params['W1'] = weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale*np.random.randn(num_filters*np.prod(input_dim[1:])/4,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    W3 = self.params['W3']
    b3 = self.params['b3']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    out,cache_conv = conv_forward_naive(X,W1,b1,conv_param)
    out,cache_relu1 = relu_forward(out)
    out,cache_maxpool = max_pool_forward_naive(out,pool_param)
    out,cache_a1 = affine_forward(out,W2,b2)
    out,cache_relu2 = relu_forward(out)
    scores,cache_a2 = affine_forward(out,W3,b3)
   
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    data_loss, dx = softmax_loss(scores,y)
    dx, dW3, db3 = affine_backward(dx,cache_a2)
    dx = relu_backward(dx,cache_relu2)
    dx, dW2, db2 = affine_backward(dx,cache_a1)
    dx = max_pool_backward_naive(dx,cache_maxpool)
    dx = relu_backward(dx,cache_relu1)
    dx, dW1, db1 = conv_backward_naive(dx, cache_conv) 
    
    dW1 += self.reg*W1
    dW2 += self.reg*W2
    dW3 += self.reg*W3
    reg_loss = 0.5*self.reg*sum(np.sum(W*W) for W in[W1,W2,W3])
    loss = data_loss + reg_loss
    grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,'W3':dW3,'b3':db3}
    
    return loss, grads
  


class Custom_net(object):
	"""
	Input: Array of dimensions(N,3,32,32)
	N = Batch size

	This class defines the following network:
	conv(32x32x32)-relu     W1, b1
	conv(32x32x32)-relu     W2, b2
	pool(32x16x16)
	conv(64x16x16)-relu     W3, b3
	conv(64x16x16)-relu     W4, b4
	pool(64x8x8)
	affine(1x1x4096)-relu   W5, b5
	affine(1x1x100)-relu    W6, b6
	affine(1x1x10)          W7, b7
	softmax
	"""
	def __init__(self, input_dim=(3, 32, 32), weight_scale=1e-3, reg=0.0,dtype=np.float32):
		self.params = {}
		self.reg = reg
		self.dtype = dtype

		#initialize weight space
		self.params['W1'] = weight_scale*np.random.randn(32,3,3,3)
		self.params['b1'] = np.zeros(32)
		self.params['W2'] = weight_scale*np.random.randn(32,32,3,3)
		self.params['b2'] = np.zeros(32)
		self.params['W3'] = weight_scale*np.random.randn(64,32,3,3)
		self.params['b3'] = np.zeros(64)
		self.params['W4'] = weight_scale*np.random.randn(64,64,3,3)
		self.params['b4'] = np.zeros(64)
		self.params['W5'] = weight_scale*np.random.randn(4096,4096)
		self.params['b5'] = np.zeros(4096)
		self.params['W6'] = weight_scale*np.random.randn(4096,100)
		self.params['b6'] = np.zeros(100)
		self.params['W7'] = weight_scale*np.random.randn(100,10)
		self.params['b7'] = np.zeros(10)

		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)
			


	def loss(self,X,y=None):

		W1 = self.params['W1']
		b1 = self.params['b1']
		W2 = self.params['W2']
		b2 = self.params['b2']
		W3 = self.params['W3']
		b3 = self.params['b3']
		W4 = self.params['W4']
		b4 = self.params['b4']
		W5 = self.params['W5']
		b5 = self.params['b5']
		W6 = self.params['W6']
		b6 = self.params['b6']
		W7 = self.params['W7']
		b7 = self.params['b7']

		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		scores = None

		# Forward Pass
		out,cache_c1 = conv_forward_naive(X,W1,b1,conv_param)
		out,cache_r1 = relu_forward(out)

		out,cache_c2 = conv_forward_naive(out,W2,b2,conv_param)
		out,cache_r2 = relu_forward(out)

		out,cache_m1 = max_pool_forward_fast(out,pool_param)

		out,cache_c3 = conv_forward_naive(out,W3,b3,conv_param)
		out,cache_r3 = relu_forward(out)

		out,cache_c4 = conv_forward_naive(out,W4,b4,conv_param)
		out,cache_r4 = relu_forward(out)

		out,cache_m2 = max_pool_forward_fast(out,pool_param)

		out,cache_a1 = affine_forward(out,W5,b5)
		out,cache_r5 = relu_forward(out)

		out,cache_a2 = affine_forward(out,W6,b6)

		out,cache_r6 = relu_forward(out)

		scores,cache_a3 = affine_forward(out,W7,b7)

		if y is None:
				return scores

		loss, grads = 0, {}

		# Calculate loss and propagate gradients
		data_loss, dx = softmax_loss(scores,y)
		dx, dW7, db7 = affine_backward(dx,cache_a3)
		dx = relu_backward(dx,cache_r6)
		dx, dW6, db6 = affine_backward(dx,cache_a2)
		dx = relu_backward(dx,cache_r5)
		dx, dW5, db5 = affine_backward(dx,cache_a1)
		dx = max_pool_backward_fast(dx,cache_m2)
		dx = relu_backward(dx,cache_r4)
		dx, dW4, db4 = conv_backward_naive(dx, cache_c4)
		dx = relu_backward(dx,cache_r3)
		dx, dW3, db3 = conv_backward_naive(dx, cache_c3)
		dx = max_pool_backward_fast(dx,cache_m1)
		dx = relu_backward(dx,cache_r2)
		dx, dW2, db2 = conv_backward_naive(dx, cache_c2)
		dx = relu_backward(dx,cache_r1)
		dx, dW1, db1 = conv_backward_naive(dx, cache_c1)

		dW1 += self.reg*W1
		dW2 += self.reg*W2
		dW3 += self.reg*W3
		dW4 += self.reg*W4
		dW5 += self.reg*W5
		dW6 += self.reg*W6
		dW7 += self.reg*W7

		reg_loss = 0.5*self.reg*sum(np.sum(W*W) for W in[W1,W2,W3,W4,W5,W6,W7])

		loss = data_loss + reg_loss
		grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,'W3':dW3,'b3':db3,'W4':dW4,'b4':db4,'W5':dW5,'b5':db5,'W6':dW6,'b6':db6,'W7':dW7,'b7':db7}

		return loss, grads















