import numpy as np
import matplotlib.pyplot as plt
from Layers.Classifiers.cnn import *
from Layers.data_utils import get_CIFAR10_data
from Layers.layers import *
from Layers.fast_layers import *
from Layers.solver import Solver


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='rmsprop',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
