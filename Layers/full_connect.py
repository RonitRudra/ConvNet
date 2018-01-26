import numpy as np

class Full_connect(object):
	"""Class Definition for a fully connected layer
	Input: An array of shape (N,d1,d2,...dn)
	Output: An array of shape(N,M)
	N = Number of instances in the batch
	D = Vectorized length of lower dimensions i.e (d1*d2*d3*......dn)
	M = Number of inputs in next layer
	"""
	def __init__(self,name,D,M):
		self.name = name
		self.W = 1e-3*np.random.randn(D,M)
		self.B = np.zeros(M)

	def forward(self,nInputPlane):
		N = nInputPlane.shape[0]
		out = nInputPlane.reshape(N,np.prod(nInputPlane.shape[1:])).dot(self.W)+self.B
		self.cache = nInputPlane
		return out

	def backward(self,dout):
		nInputPlane = self.cache
		dx, dw, db = None, None, None
		N = nInputPlane.shape[0]
		dx = np.dot(dout,self.W.T).reshape(nInputPlane.shape)
		dw = nInputPlane.reshape(N,np.prod(nInputPlane.shape[1:])).T.dot(dout)
		db = np.sum(dout,axis=0)
		self.dw = dw
		self.db = db
		#Use dw and db to update the FC layer weights and biases 
		return dx

