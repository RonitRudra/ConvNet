import numpy as np
class Optimize_rmsprop(object):
  """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
  def __init__(self,n,learning_rate=1e-2,decay_rate=0.99,epsilon=1e-8):
    self.cache = np.zeros_like(n)
    self.lr = learning_rate
    self.dr = decay_rate
    self.eps = epsilon
  
  def update_rmsprop(self,x, dx):
    next_x = None
    self.cache = self.dr * self.cache + (1 - self.dr) * dx**2
    grad_scale = np.sqrt(self.cache)+self.eps
    update = - self.lr*dx/grad_scale
    next_x = x + update

    params_scaled = np.linalg.norm(x.ravel())
    update_scaled = np.linalg.norm(update.ravel())
    print 'update/parameter ratio: %f' % (update_scaled/params_scaled)
    return next_x