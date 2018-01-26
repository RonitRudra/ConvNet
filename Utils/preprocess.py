import numpy as np
from os import listdir
import cPickle

def read_data(path):
	files = listdir(path)
	data = np.zeros(shape=(1,3072))
	labels = np.zeros(shape=(1,1))

	for i in files:
		batch = unpickle(path+i)
		data = np.concatenate((data, np.array(batch['data'])), axis=0)
		labels = np.concatenate((labels, np.array(batch['labels']).reshape(-1,1)), axis=0)

	data = data[1:,:]
	labels = labels[1:,:].astype(int)
	return [data,labels]

def unpickle(files):
	fo = open(files, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def split(data,labels,num_training,num_validation,num_dev):

	#----------------------Validation-----------------------------------------
	mask = range(num_training, num_training + num_validation)
	data_val = data[mask]
	labels_val = labels[mask]
	#----------------------Train----------------------------------------------
	mask = range(num_training)
	data_train = data[mask]
	labels_train = labels[mask]
	labels_train = labels_train.reshape(-1,)
	#-----------------------Development---------------------------------------
	mask = np.random.choice(num_training, num_dev, replace=False)
	data_dev = data_train[mask]
	label_dev = labels_train[mask]
	#-------------------------------------------------------------------------

	#-----------------------Preprocessing-------------------------------------
	mean_image = np.mean(data_train,axis=0)
	data_train -= mean_image
	data_val -= mean_image
	data_dev -= mean_image
	data_train = np.hstack([data_train, np.ones((data_train.shape[0], 1))])
	data_val = np.hstack([data_val, np.ones((data_val.shape[0], 1))])
	data_dev = np.hstack([data_dev, np.ones((data_dev.shape[0], 1))])
	#-------------------------------------------------------------------------
	return data_train,data_val,data_dev,labels_train,labels_val,label_dev

def reshaping(data,dim1,dim2,dim3):
	N = data.shape[0]
	reshaped = np.zeros((N,dim1,dim2,dim3))
	res = dim2*dim3
	for i in xrange(N):
		img = data[i,:]
		img_r, img_g, img_b = img[0:res].reshape(dim2,dim3), img[res:(2*res)].reshape(dim2,dim3), img[(2*res):(3*res)].reshape(dim2,dim3)
		img = np.array((img_r,img_g,img_b))
		reshaped[i,:,:,:] = img
	return reshaped