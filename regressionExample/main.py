import os
import numpy as np
import pandas as pd
from GradientDescent import GradientDescent
from NormalEquation import NormalEquation
from Plotter import Plot

ITERATIONS = 1500
ALPHA = 0.01

## Plots ##

def plot_initial_data():
    graph = Plot(x_values=None, function=None,
                         title="Predicted Profit vs. Population Size",
                         x_label="Population",
                         y_label="Profit")
    graph.scatter(data.Population, data.Profit, label="Traning Data")
    graph.show()

def plot_cost_function_over_iterations(x_data, function):
    graph = Plot(x_values=x_data, function=function,
                 title="Cost Function Over Iterations",
                 x_label="Iterations",
                 y_label="J (θ0, θ1)")
    graph.show()


def plot_hypothesis_without_predicted_value(x_data, function):
    graph = Plot(x_values=x_data, function=function,
                         title="Predicted Profit vs. Population Size",
                         x_label="Population",
                         y_label="Profit")
    graph.scatter(data.Population, data.Profit, label="Traning Data")
    graph.show()

def plot_hypothesis_with_predicted_value(x_data, function, value, predicted_value):
    graph = Plot(x_values=x_data, function=function,
                         title="Predicted Profit vs. Population Size",
                         x_label="Population",
                         y_label="Profit")
    graph.scatter(data.Population, data.Profit, label="Traning Data")
    graph.scatter(value, predicted_value, label="Predicted Value", c="g")
    graph.text(value - 0.55, predicted_value + 1, str(predicted_value)[0:6])
    graph.show()

def menu_algorithm():
    print("\n---")
    print("Choose your regression algorithm:")
    print("\n1. Gradient Descent")
    print("2. Normal Equation")
    return int(input(("\nChoose (1, 2): ")))

def choose_algorithm():
    option = menu_algorithm()
    if option == 1:
        return GradientDescent(ALPHA, ITERATIONS)
    elif option == 2:
        return NormalEquation(ITERATIONS)

## Main ##

if __name__=="__main__":

    # Load dataset
    data_path = os.getcwd()+"/data/ex1data1.txt"

    # Create Pandas Data frame with dataset
    data = pd.read_csv(data_path, sep=",", header=None, names=["Population", "Profit"], dtype=np.float)

    # Plot Initial Data
    plot_initial_data()

    # Insert column of ones to fill matrix size (1 is neutral in multiplications)
    ones = np.ones(data.__len__())
    data.insert(loc=0, column='Ones', value=ones)

    # Separate X and Y data frames
    x_array = data[["Ones", "Population"]]
    y_output = data[["Profit"]]

    m_size, n_categories = x_array.shape

    # Choose algorithm between Normal Equation and Gradient Descent
    algorithm = choose_algorithm()

    theta = np.ones((n_categories, 1))
    theta, j_gradients = algorithm.get_theta(theta, x_array, y_output)
    print("theta found:",theta)

    # Calculate function
    x_data = np.linspace(data.Population.min(), data.Population.max())
    function = theta[0] + (theta[1] * x_data)

    # Create graph with Function plotted
    plot_hypothesis_without_predicted_value(x_data, function)

    # Create graph with cost function over iterations (only for Gradient Descent)
    if (type(algorithm) is GradientDescent):
        plot_cost_function_over_iterations(np.arange(ITERATIONS), j_gradients)

    while True:
        valueToPredict = int(input("Value  to predict: "))
        predicted_value = theta[0] + (theta[1] * valueToPredict)
        print("Predicted value: ", str(predicted_value)[0:5])
        plot_hypothesis_with_predicted_value(x_data, function, valueToPredict, predicted_value)



