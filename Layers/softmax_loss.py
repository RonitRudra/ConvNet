import numpy as np

class Softmax_loss():
	"""Forward and Backward API for Softmax Loss"""
	def __init__(self, name):
		self.name = name

	def forward(self,nInputPlane,labels):
		probs = np.exp(nInputPlane - np.max(x,axis = 1, keepdims= TRUE))
		probs /= np.sum(probs,axis = 1, keepdims = TRUE)
		N = nInputPlane.shape[0]
		loss = -np.sum(np.log(probs[np.arange(N),labels]))/N
		self.cache = probs,N,labels
		return loss

	def backward():
		probs,N,labels = self.cache
		dx = probs.copy()
		dx[np.arange(N),labels] -=1
		dx /=N
		return dx
