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

        self.label = rows[0]
        self.data = torch.tensor(np.asarray(rows[1:], dtype = float), dtype = torch.float32)

        # print(self.data)
        # self.input_label = label[:1]
        # self.output_label = label[1:]

        # self.split_data(0.5)
        # print(self.train)
        # print(self.val)
        

    def make_nn(self, model):
        self.model = model

    def split_data(self, ratio, col_split):  #split the data into two subsets, ratio = ratio of train to val
        DT = self.data[torch.randperm(self.data.size()[0])]
        l = int(len(self.data)*ratio)
        self.train = DT[:l]
        self.val = DT[l:]

        self.input_label = self.label[:col_split]
        self.output_label = self.label[col_split:]

        self.x_train = self.train.mT[:col_split].mT
        self.y_train = self.train.mT[col_split:].mT
        self.x_val = self.val.mT[:col_split].mT
        self.y_val = self.val.mT[col_split:].mT

    def grad_descent(self):
        

T = Train("Linear Predictor/FFN/train.csv")
T.read_file()
T.split_data(0.5, 1)

model = nn.Sequential(
    nn.Linear(1, 1)
)

T.make_nn(model)
T.grad_descent()

