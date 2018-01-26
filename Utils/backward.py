def backward():
	# Backward Pass
	dout = SVM.backward()
	dout = F1.backward(dout)
	# Update weights of F1 here
	F1.W = F1_w_u.update_rmsprop(F1.W,F1.dw)
	F1.B = F1_b_u.update_rmsprop(F1.B,F1.db)
	#
	dout = M1.backward(dout)
	dout = R3.backward(dout)
	dout = C3.backward(dout)
	# Update weights of C3 here
	C3.H = C3_h_u.update_rmsprop(C3.H,C3.dh)
	C3.B = C3_b_u.update_rmsprop(C3.B,C3.db)
	#
	dout = R2.backward(dout)
	dout = C2.backward(dout)
	# Update weights of C2 here
	C2.H = C2_h_u.update_rmsprop(C2.H,C2.dh)
	C2.B = C2_b_u.update_rmsprop(C2.B,C2.db)
	#
	dout = R1.backward(dout)
	dout = C1.backward(dout)
	# Update weights of C1 here
	C1.H = C1_h_u.update_rmsprop(C1.H,C1.dh)
	C1.B = C1_b_u.update_rmsprop(C1.B,C1.db)