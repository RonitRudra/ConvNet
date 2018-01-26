import cPickle

def save(objname,filename):
	files = open('Objects/'+filename+'.obj','w')
	cPickle.dump(objname,files)

def load(filename):
	files = open('Objects/'+filename+'.obj','r')
	x = cPickle.load(files)
	return x