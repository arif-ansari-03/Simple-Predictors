# divide the training dataset into train and validation sets and then train using

import csv
import torch.nn as nn
import torch
import numpy as np

class Train:
    def __init__(self, data_file):
        self.data_file = data_file

    def read_file(self):  # read file and split the columns into input columns and output columns (currently only works for one input column)
        rows = []
        with open(self.data_file, newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                rows.append(row)

        label = rows[0]
        data = torch.tensor(np.asarray(rows[1:], dtype = float))

        self.input_label = label[:1]
        self.output_label = label[1:]

        input_data = data.T[:1]
        output_data = data.T[1:]

        self.input_data = input_data.T
        self.output_data = output_data.mT

        print(self.input_label)
        print(self.input_data)

        print(self.output_label)
        print(self.output_data)

    def make_nn(num_input, num_output):
        self.model = nn.sequential
        (
            nn.Linear(num_input, num_output)
        )

    def split_data(self):  #split the data into two subset objects
        DT = torch.utils.data.random_split(self.output_data, [0.5, 0.5])
        self.x_train, self.x_val = torch.tensor(DT) #use torch.tensor(<subset object>) to convert into objects
        print(self.x_train)
        print(self.x_val)  # IMPORTANT TODO: SPLIT DATA BEFORE SPLITTING INTO INPUT AND OUTPUT DATA

T = Train("Linear Predictor/FFN/train.csv")
T.read_file()
T.split_data()