def forward(x):
	#Forward Pass
	out = C1.forward(x)
	out = R1.forward(out)
	out = C2.forward(out)
	out = R2.forward(out)
	out = C3.forward(out)
	out = R3.forward(out)
	out = M1.forward(out)
	out = F1.forward(out)
	return out