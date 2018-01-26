import numpy as np
class Optimize_adam(object):
	"""
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
	def __init__(self,n,learning_rate=1e-3,beta1=0.9,beta2=0.99,epsilon=1e-8):
		self.lr = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = epsilon
		self.m = np.zeros_like(n)
		self.v = np.zeros_like(n)
		self.t = 0

	def update_adam(self,x,dx):
		next_x = None
		self.m = self.beta1*self.m + (1-self.beta1)*dx
		self.v = self.beta2*self.v + (1-self.beta2)*(dx**2)
		update = - self.lr*self.m/(np.sqrt(self.v)+self.eps)
		next_x = x + update

		params_scaled = np.linalg.norm(x.ravel())
		update_scaled = np.linalg.norm(update.ravel())
		print update_scaled/params_scaled
		return next_x


		