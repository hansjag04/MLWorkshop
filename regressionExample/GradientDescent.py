import ThetaCalculator as tc
import numpy as np


class GradientDescent(tc.ThetaCalculator):

    alpha = 0.01
    iterations = 1

    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations

    def get_theta(self, theta, x_training_set, y_output):
        m_size = len(x_training_set)
        x_transpose = x_training_set.transpose()

        # Costs in all iterations
        j_costs = np.zeros(self.iterations)

        for i in range(self.iterations):
            # Calculate hypothesis
            hypothesis = self.calculate_hypothesis(theta, x_training_set)

            # Save cost
            j_costs[i] = self.calculate_cost(hypothesis, m_size, y_output)

            # Update theta
            gradient = self.calculate_gradient(hypothesis, m_size, x_transpose, y_output)
            theta = theta - self.alpha * gradient

        return theta, j_costs

    def calculate_hypothesis(self, theta, x_training_set):
        return np.dot(x_training_set, theta)

    def calculate_gradient(self, hypothesis, m_size, x_transpose, y_output):
        return  np.dot(x_transpose, hypothesis - y_output) / m_size

    def calculate_cost(self, hypothesis, m_size, y_output):
        return np.sum((hypothesis - y_output) ** 2) / (2 * m_size)
