# generate the training and testing dataset for the lines given parameters
import numpy as np
import csv


# simple line in xy plane

class XYLine:
    def __init__(self, a, b):  # y = ax + b
        self.a = a
        self.b = b

    #gen_data(number of data points, range of x as a tuple (lower bound, upper bound) both inclusive)
    def gen_X(self, num_data, x_range): 
        X = np.random.random(num_data)
        X = (X * (x_range[1]-x_range[0])) + x_range[0]
        self.X = X

    def write_data(self, file_name):
        with open(file_name, 'w', newline = '') as csv_file:
            file_writer = csv.writer(csv_file)
            for x in self.X:
                file_writer.writerow([x, self.a*x+self.b])

    def gen_data(self, num_data, x_range, file_name):
        self.gen_X(num_data, x_range)
        self.write_data(file_name)


XYLine(5, 0).gen_data(4, (1, 4), "Line Predictor/FFN/train.csv")
        