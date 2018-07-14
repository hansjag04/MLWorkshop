import ThetaCalculator as tc
import numpy as np

class NormalEquation(tc.ThetaCalculator):
    def __init__(self, iterations):
        self.iterations = iterations

    def get_theta(self, theta, x_array, y_array):
        return [1, 1], np.ones(shape=self.iterations)