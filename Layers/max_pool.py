import numpy as np
class Max_pool(object):
	"""docstring for ClassName"""
	def __init__(self,name,h_pool,w_pool,stride):
		self.name = name
		self.h_pool = h_pool
		self.w_pool = w_pool
		self.S = stride

	def forward(self,nInputPlane):
		[xB,xD,xH,xW] = nInputPlane.shape
		outH = 1 + (xH-self.h_pool)/self.S
		outW = 1 + (xW-self.w_pool)/self.S
		out = np.zeros((xB,xD,outH,outW))

		for i in xrange(xB):
			for j in xrange(xD):
				for k in xrange(outH):
					hs = k*self.S
					for l in xrange(outW):
						ws = l*self.S
						window = nInputPlane[i,j,hs:hs+self.h_pool,ws:ws+self.w_pool]
						out[i,j,k,l] = np.max(window)

		self.cache = nInputPlane
		return out

	def backward(self,dout):
		nInputPlane = self.cache
		[xB,xD,xH,xW] = nInputPlane.shape
		outH = 1 + (xH-self.h_pool)/self.S
		outW = 1 + (xW-self.w_pool)/self.S

		dx = np.zeros_like(nInputPlane)

		for i in xrange(xB):
			for j in xrange(xD):
				for k in xrange(outH):
					hs = k*self.S
					for l in xrange(outW):
						ws=l*self.S
						window = nInputPlane[i,j,hs:hs+self.h_pool,ws:ws+self.w_pool]
						m = np.max(window)
						dx[i,j,hs:hs+self.h_pool,ws:ws+self.w_pool] += (window==m)*dout[i,j,k,l]

		return dx

		