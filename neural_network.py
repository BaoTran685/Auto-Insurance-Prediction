import numpy as np

def init_params():
  def he_init(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
  
  def xavier_init(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

  W1 = he_init(19, 16)
  b1 = np.zeros((1, 16))
  W2 = he_init(16, 1)
  b2 = np.zeros((1, 1))
  return W1, b1, W2, b2


def ReLU(Z):
  return np.maximum(Z, 0)

def ReLU_derivative(Z):
  return Z > 0

def Sigmoid(Z):
  return 1 / (1 + np.exp(-Z))

def Sigmoid_derivative(Z):
	return Sigmoid(Z) * (1 - Sigmoid(Z))

def compute_loss(A, y):
  # finding the loss
  diff = (A - y) ** 2
  m, _ = A.shape
  return np.sum(diff) / m



class Neural_Network:
  def __init__(self, alpha, decay_rate, iterations):
    self.alpha = alpha
    self.decay_rate = decay_rate
    self.iterations = iterations
    self.W1, self.b1, self.W2, self.b2 = init_params()
  
  def print_all(self):
    print(self.alpha, self.decay_rate, self.iterations, self.W1, self.b1, self.W2, self.b2)

  def forward_prop(self, X):
    # layer 1
    Z1 = np.dot(X, self.W1) + self.b1
    A1 = ReLU(Z1)
    # layer 2
    Z2 = np.dot(A1, self.W2) + self.b2
    A2 = Sigmoid(Z2)
    return Z1, A1, Z2, A2
  
  def backward_prop(self, Z1, A1, Z2, A2, X, Y):
    m, _ = Y.shape
    # layer 2
    dZ2 = 2 / m * (A2 - Y) * Sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis = 0, keepdims = True)
    # layer 1
    dZ1 = np.dot(dZ2, self.W2.T) * ReLU_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis = 0, keepdims = True)
    return dW1, db1, dW2, db2
  
  def update_params(self, dW1, db1, dW2, db2):
    self.W1 -= self.alpha * dW1
    self.b1 -= self.alpha * db1
    self.W2 -= self.alpha * dW2
    self.b2 -= self.alpha * db2

  def gradient_descent(self, X, Y):
    last_loss = 1000000 # init to some large value
    for i in range(self.iterations):
      Z1, A1, Z2, A2 = self.forward_prop(X)
      dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
      self.update_params(dW1, db1, dW2, db2)
      if (i % 100 == 0):
        print(f"Iterations: {i}")
        loss = compute_loss(A2, Y)
        print(f"Loss: {loss}")
        # if (loss >= last_loss and (last_loss - loss) / loss <= 0.001):
        #   print("Stop")
        #   break
        last_loss = loss
      # update the learning rate
      self.alpha = self.alpha * np.exp(-self.decay_rate)

  def mini_batch_gradient_descent(self, batch_size, X, Y):
    number_of_samples, _ = X.shape
    for i in range(self.iterations):
      permutation = np.random.permutation(number_of_samples)
      X_shuffle = X[permutation]
      Y_shuffle = Y[permutation]
      # loop through the data set with increment of batch_size
      for j in range(0, number_of_samples, batch_size):
        # get the current batch
        X_batch = X_shuffle[j: j + batch_size]
        Y_batch = Y_shuffle[j: j + batch_size]
        # perform
        Z1, A1, Z2, A2 = self.forward_prop(X_batch)
        dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X_batch, Y_batch)
        self.update_params(dW1, db1, dW2, db2) # update the params for every data of size batch_size
      if (i % 100 == 0):
        _, _, _, A2 = self.forward_prop(X)
        print(f"Iterations: {i}")
        loss = compute_loss(A2, Y)
        print(f"Loss: {loss}")
      # update the learning rate
      self.alpha = self.alpha * np.exp(-self.decay_rate)