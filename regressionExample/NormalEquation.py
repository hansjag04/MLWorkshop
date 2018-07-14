import ThetaCalculator as tc
import numpy as np

class NormalEquation(tc.ThetaCalculator):
    def __init__(self, iterations):
        self.iterations = iterations

    ## Function: ( X_transpose * X )^-1  *  X_transpose * Y
    def get_theta(self, theta, x_training_set, y_output):
        x_transpose = x_training_set.T
        xt_x = np.dot(x_transpose, x_training_set)  # X_transpose * X
        xt_y = np.dot(x_transpose, y_output)        # X_transpose * Y
        normal_eq = np.dot(np.linalg.inv(xt_x), xt_y)

        return normal_eq, np.ones(shape=self.iterations)