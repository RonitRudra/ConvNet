import numpy as np

class SVM_loss(object):
	"""Forward and Backward API for SVM loss function"""
	def __init__(self,name):
		self.name = name

	def forward(self, nInputPlane,labels):
		N = nInputPlane.shape[0]
		correct_score = nInputPlane[np.arange(N),labels]
		margins = np.maximum(0,nInputPlane-correct_score[:,np.newaxis]+1)
		margins[np.arange(N),labels] = 0
		loss = np.sum(margins)/N
		self.cache = (margins,nInputPlane,N,labels)
		return loss

	def backward(self):
		[margins,nInputPlane,N,labels] = self.cache
		num_pos = np.sum(margins>0,axis=1)
		dx = np.zeros_like(nInputPlane)
		dx[margins>0] = 1
		dx[np.arange(N),labels] = -num_pos*dx[np.arange(N),labels]
		dx /=N
		return dx

	def predict(self,nInputPlane,Labels):
		lab_pred = []
		for i in xrange(Labels.shape[0]):
			lab_pred.append(np.argmax(nInputPlane,axis=1))
			lab_pred = np.hstack(lab_pred)

		accuracy = np.mean(y_pred == Labels)
		return accuracy




