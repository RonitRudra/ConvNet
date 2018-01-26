class Relu(object):
	"""Rectified Linear Unit object

	"""
	def __init__(self,name):
		self.name = name
	
	def forward(self,incoming):
		out = incoming*(incoming>0)
		self.cache = incoming
		return out

	def backward(self,dout):
		incoming = self.cache
		dx = dout*(incoming>=0)
		return dx

		

