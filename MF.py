import numpy as np

class MF():

    def __init__(self, data, K_feature, beta, lambda_value, iterations):
      """
      - R: user-item rating matrix
      - K_feature: number of latent dimensions
      - beta: learning rate
      - lambda_value: regularization parameter
      """

      self.data = data
      self.num_users, self.num_items = data.shape
      self.K_feature = K_feature
      self.beta = beta
      self.lambda_value = lambda_value
      self.iterations = iterations

    def train(self):
      # Initialize user and item latent feature matrice
      self.P = np.random.normal(scale=1./self.K_feature, size=(self.num_users, self.K_feature))
      self.Q = np.random.normal(scale=1./self.K_feature, size=(self.num_items, self.K_feature))
      # Create a list of training samples
      self.samples = [
          (i, j, self.data[i, j])
          for i in range(self.num_users)
          for j in range(self.num_items)
          if self.data[i, j] > 0
      ]
      # Perform gradient descent for number of iterations
      for i in range(self.iterations):
          np.random.shuffle(self.samples)
          self.gradient_descent()

    def gradient_descent(self):
      for i, j, r in self.samples:
          # Computer prediction and error
          prediction = self.P[i, :].dot(self.Q[j, :].T)
          e = (r - prediction)
          self.P[i, :] += self.beta * (e * self.Q[j, :] - self.lambda_value * self.P[i,:])
          self.Q[j, :] += self.beta * (e * self.P[i, :] - self.lambda_value * self.Q[j,:])
      # Return new matrix with predicted data
      return self.P.dot(self.Q.T)