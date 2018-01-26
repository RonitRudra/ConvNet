import numpy as np
'''
x is an n dimensional array of (N,d1,d2,.....dn)
y is an array of size N
'''
def train_acc(x,y):
	class_winner = np.argmax(x,axis = 1)
	acc = np.mean(class_winner==y)
	print "train accuracy: %f" % (acc)

def val_acc(x,y):
	out = forward(x)
	class_winner = np.argmax(out,axis = 1)
	acc = np.mean(class_winner==y)
	print "validation accuracy: %f" % (acc)


