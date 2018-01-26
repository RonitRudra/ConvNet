#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#----------------------------MAIN SCRIPT FOR CONVOLUTION NEURAL NETWORK----------------------------

# 1.-------------------------Module, Function and Class Import-------------------------------------
from os import system
import numpy as np
from Layers.conv import Conv
from Layers.relu import Relu
from Layers.max_pool import Max_pool
from Layers.full_connect import Full_connect
from Layers.svm_loss import SVM_loss
from Update.rmsprop import Optimize_rmsprop
from Update.adam import Optimize_adam
from Utils.preprocess import read_data,unpickle,split,reshaping
from Utils.pickling import save,load
from Utils.forward import forward
from Utils.backward import backward
from Utils.accuracy import *

#==================================================================================================

# 2.------------------------Preprocessing Data-----------------------------------------------------

path_train = "/home/ronit/Documents/Datasets/cifar-10-batches-py/Train/"
[data,labels] = read_data(path_train)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_training = input("Enter Train Split: ")
num_validation = data.shape[0] - num_training
num_dev = input("Enter Development Split: ")
[data_train,data_val,data_dev,labels_train,labels_val,label_dev] = split(data,labels,num_training,num_validation,num_dev)

data_train = reshaping(data_train,3,32,32)
data_val = reshaping(data_val,3,32,32)
data_dev = reshaping(data_dev,3,32,32)

# 3.------------------------Define Layers------------------------------------------------------------
C1 = Conv('C1',64,3,3,1,1)#Nx64x32x32
C1_h_u = Optimize_rmsprop(C1.H)
C1_b_u = Optimize_rmsprop(C1.B)

C2 = Conv('C2',64,64,3,1,1)#Nx64x32x32
C2_h_u = Optimize_rmsprop(C2.H)
C2_b_u = Optimize_rmsprop(C2.B)

P1 = Max_pool('P1', 2, 2, 2)#Nx64x316x16

C3 = Conv('C3',128,64,3,1,1)#Nx128x16x16
C3_h_u = Optimize_rmsprop(C3.H)
C3_b_u = Optimize_rmsprop(C3.B)

C4 = Conv('C4',128,128,3,1,1)#Nx128x16x16
C4_h_u = Optimize_rmsprop(C4.H)
C4_b_u = Optimize_rmsprop(C4.B)

P2 = Max_pool('P2', 2, 2, 2)#Nx128x8x8

C5 = Conv('C5',256,128,3,1,1)#Nx256x8x8
C5_h_u = Optimize_rmsprop(C5.H)
C5_b_u = Optimize_rmsprop(C5.B)

C6 = Conv('C6',256,256,3,1,1)#Nx256x8x8
C6_h_u = Optimize_rmsprop(C6.H)
C6_b_u = Optimize_rmsprop(C6.B)

C7 = Conv('C7',256,256,3,1,1)#Nx256x8x8
C7_h_u = Optimize_rmsprop(C7.H)
C7_b_u = Optimize_rmsprop(C7.B)

P3 = Max_pool('P3',2,2,2)#Nx256x4x4

FC1 = Full_connect('FC1',4096,4096)#Nx4096
FC1_w_u = Optimize_rmsprop(FC1.W)
FC1_b_u = Optimize_rmsprop(FC1.B)

FC2 = Full_connect('FC2',4096,4096)#Nx4096
FC2_w_u = Optimize_rmsprop(FC2.W)
FC2_b_u = Optimize_rmsprop(FC2.B)

FC3 = Full_connect('FC3',4096,10)#Nx10
FC3_w_u = Optimize_rmsprop(FC3.W)
FC3_b_u = Optimize_rmsprop(FC3.B)
SVM = SVM_loss('SVM')


#----------------------------------------------------------------------------------------------------
choice = input("Choose Training(1) or Development(2)?: ")
if(choice == 1):
	x = data_train
	y = labels_train
elif(choice== 2):
	x = data_dev
	y = label_dev
else:
	print("Enter Valid Choice")

batch_size = input("Set the Batch Size: ")
batch_index = range(0,x.shape[0],batch_size)
num_batch = x.shape[0]/batch_size
epochs = input("Set the number of epochs: ")
loss_cache = []
k = 1

for i in xrange(epochs):
	for j in batch_index:
		x_in = x[j:j+batch_size,:,:,:]
		y_in = y[j:j+batch_size]
		print "Epoch: %d  Batch: %d/%d" % (i,k,num_batch)
		k+=1
		
		# Forward Pass-----------------------------------------------------------------------------
		out = C1.forward(x_in)
		out = C2.forward(out)
		out = P1.forward(out)
		out = C3.forward(out)
		out = C4.forward(out)
		out = P2.forward(out)
		out = C5.forward(out)
		out = C6.forward(out)
		out = C7.forward(out)
		out = P3.forward(out)
		out = FC1.forward(out)
		out = FC2.forward(out)
		out = FC3.forward(out)
		#------------------------------------------------------------------------------------------
		# Print train accuracy
		train_acc(x_in, y_in)

		# Loss Determination
		loss = SVM.forward(out,y_in)
		print("loss is %f") % loss
		loss_cache.append(loss)
		if(j==batch_size-1):
			val_acc(data_val,labels_val)


		# Backward Pass----------------------------------------------------------------------------
		dout = SVM.backward()
		#
		dout = FC3.backward(dout)
		# Update weights of FC3 here
		FC3.W = FC3_w_u.update_rmsprop(FC3.W,FC3.dw)
		FC3.B = FC3_b_u.update_rmsprop(FC3.B,FC3.db)
		#
		dout = FC2.backward(dout)
		# Update weights of FC2 here
		FC2.W = FC2_w_u.update_rmsprop(FC2.W,FC2.dw)
		FC2.B = FC2_b_u.update_rmsprop(FC2.B,FC2.db)
		#
		dout = FC1.backward(dout)
		# Update weights of FC1 here
		FC1.W = FC1_w_u.update_rmsprop(FC1.W,FC1.dw)
		FC1.B = FC1_b_u.update_rmsprop(FC1.B,FC1.db)
		#
		dout = P3.backward(dout)
		#
		dout = C7.backward(dout)
		# Update weights of C7 here
		C7.H = C7_h_u.update_rmsprop(C7.H,C7.dh)
		C7.B = C7_b_u.update_rmsprop(C7.B,C7.db)
		#
		dout = C6.backward(dout)
		# Update weights of C6 here
		C6.H = C6_h_u.update_rmsprop(C6.H,C6.dh)
		C6.B = C6_b_u.update_rmsprop(C6.B,C6.db)
		#
		dout = C5.backward(dout)
		# Update weights of C5 here
		C5.H = C5_h_u.update_rmsprop(C5.H,C5.dh)
		C5.B = C5_b_u.update_rmsprop(C5.B,C5.db)
		#
		dout = P2.backward(dout)
		#
		dout = C4.backward(dout)
		# Update weights of C4 here
		C4.H = C4_h_u.update_rmsprop(C4.H,C4.dh)
		C4.B = C4_b_u.update_rmsprop(C4.B,C4.db)
		#
		dout = C3.backward(dout)
		# Update weights of C3 here
		C3.H = C3_h_u.update_rmsprop(C3.H,C3.dh)
		C3.B = C3_b_u.update_rmsprop(C3.B,C3.db)
		#
		dout = P1.backward(dout)
		#
		dout = C2.backward(dout)
		# Update weights of C2 here
		C2.H = C2_h_u.update_rmsprop(C2.H,C2.dh)
		C2.B = C2_b_u.update_rmsprop(C2.B,C2.db)
		#
		dout = C1.backward(dout)
		# Update weights of C1 here
		C1.H = C1_h_u.update_rmsprop(C1.H,C1.dh)
		C1.B = C1_b_u.update_rmsprop(C1.B,C1.db)

		#------------------------------------------------------------------------------------------

loss_cache = np.array(loss_cache)
