import matplotlib.pyplot as plt

class Plot():

    def __init__(self, x_values, function, title, x_label, y_label):
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        if(x_values is not None):
            self.ax.plot(x_values, function, "r", label="")
        self.ax.legend(loc=2)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)

    def text(self, x, y, text):
        self.ax.text(x, y, text, bbox=dict(facecolor="g"))

    def scatter(self, x_var, y_var, label, c=None):
        self.ax.scatter(x_var, y_var, c=c, label=label)

    def show(self):
        plt.show()
