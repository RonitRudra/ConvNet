import numpy as np
from scipy.signal import convolve2d

class Conv(object):
	"""Convolutional Layer with Forward and Backward API Calls
	filters = number of filters to be used in the layer
	stride = number of pixels between two consecutive windows, assuming equal in both axis
	padding = amount of zero padding to be done in both axis
		Forward API
		nInputPLane is a tensor input with dimensions [batch,depth,row_size,column_size]
		Total parameters = F*F*xD*K plus K biases
		The output would be the activation map
		Backward API
		Takes in the cached input during the forward pass and the upstream gradients
	"""
	def __init__(self,name,filters,channels,kernels,stride,padding):
		self.name = name
		self.K = filters
		self.F = kernels
		self.C = channels
		self.S = stride
		self.P = padding
		# Initialize weight tensor
		self.H = 1e-3*np.random.randn(self.K,self.C,self.F, self.F)
		self.B = np.zeros(self.K)
		

	def forward(self,nInputPlane):
		##TODO: add stride in convolution
		##TODO: add bias
		[xB,xD,xH,xW] = nInputPlane.shape
		# Set dimensions of activation map such that spatial dimension is maintained
		outW = (xW - self.F + 2*self.P)/self.S + 1
		outH = (xH - self.F + 2*self.P)/self.S + 1
		outD = self.K
		# Create Empty frame of activation map
		out = np.zeros((xB,outD,outH,outW))
		# Pad the input images
		# npad is a tuple of (n_before, n_after) for each dimension
		npad = [(0,0),(0,0),(self.P,self.P),(self.P,self.P)]
		# padding only the width and height dimensions
		x_pad = np.pad(nInputPlane,npad,mode='constant')

		for i in xrange(xB):
			for j in xrange(self.K):
				for k in xrange(outH):
					hs = k*self.S
					for l in xrange(outW):
						ws = l*self.S
						window = x_pad[i,:,hs:hs+self.F,ws:ws+self.F]
						out[i,j,k,l] = np.sum(window*self.H[j]) + self.B[j]			
		
		self.cache = (nInputPlane,xB,xD,xH,xW,npad)
		return out
		


	def backward(self,dout):
		[hF,hD,hH,hW] = self.H.shape
		[nInputPlane,xB,xD,xH,xW,npad] = self.cache
		outH = 1 + (xH + 2 * self.P - hH)/self.S
		outW = 1 + (xW + 2 * self.P - hW) / self.S

		dx = np.zeros_like(nInputPlane)
		dh = np.zeros_like(self.H)
		db = np.zeros_like(self.B)
		dx_pad = np.pad(dx,npad, mode = 'constant')
		x_pad = np.pad(nInputPlane,npad, mode = 'constant')

		for i in xrange(xB):
			for j in xrange(self.K):
				for k in xrange(outH):
					hs = k*self.S
					for l in xrange(outW):
						ws = l*self.S
						window = x_pad[i,:,hs:hs+self.F,ws:ws+self.F]
						db[j] = db[j] + dout[i,j,k,l]
						dh[j] = dh[j] + window*dout[i,j,k,l]
						dx_pad[i,:,hs:hs+self.F,ws:ws+self.F] += self.H[j]*dout[i,j,k,l]

		dx = dx_pad[:,:,self.P:self.P+xH,self.P:self.P+xW]
		self.dh = dh
		self.db = db
		#Use dh and db to update the Conv layer weights and biases
		return dx








		


