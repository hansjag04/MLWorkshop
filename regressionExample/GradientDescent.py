import ThetaCalculator as tc
import numpy as np


class GradientDescent(tc.ThetaCalculator):

    alpha = 0.01
    iterations = 1

    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations

    def get_theta(self, theta, x_training_set, y_output):
        m_size = x_training_set.shape[0]
        x_transpose = x_training_set.transpose()

        # Costs in all iterations
        j_costs = np.zeros(self.iterations)

        for i in range(self.iterations):
            # Calculate hypothesis
            hypothesis = self.calculate_hypothesis(theta, x_training_set)
            error = hypothesis - y_output

            # Save cost
            j_costs[i] = self.calculate_cost(error, m_size)

            # Update theta
            gradient = self.calculate_gradient(x_transpose, error, m_size)
            theta = theta - self.alpha * gradient

        return theta, j_costs

    def calculate_hypothesis(self, theta, x_training_set):
        return np.dot(x_training_set, theta)

    def calculate_gradient(self, x_transpose, error, m_size):
        return  np.dot(x_transpose, error) / m_size

    # Loss Cost function J(theta) = 1/2m * (SUM[i=0, m](h(X[i]) - Y[i])**2)
    def calculate_cost(self, error, m_size):
        return np.sum(error ** 2) / (2 * m_size)
