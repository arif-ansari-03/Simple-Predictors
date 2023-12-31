# divide the training dataset into train and validation sets and then train using

import csv
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time

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
        

    def make_nn(self, model):
        self.best_model = model.state_dict()
        self.best_mse = 100000
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

    def grad_descent(self, num_epochs, batch_size, loss_fn, optimizer):
        batch_range = torch.arange(0, len(self.x_train), batch_size)

        history = []

        for epoch in range(1,num_epochs+1):
            # tqdm() -> first arg is like (range(a, b)), remaining are loading bar descriptions
            # unit -> x s/unit means x seconds taken on one unit, mininterval = time btw updates, disable -> something for wrapper?
            with tqdm.tqdm(batch_range, unit = "batch", mininterval = 0, disable = False) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    x_batch = self.x_train[i:i+batch_size]
                    y_batch = self.y_train[i:i+batch_size]
                    y_pred = self.model(x_batch)  # y_pred has info abt it's model 
                    loss = loss_fn(y_pred, y_batch) # now loss has info abt the cost and y_pred's model            
                    
                    # using info abt the model & the cost, loss.backward() can now compute gradients
                    # but b4 that, we set the gradients to zero, so that it doesn't add to old gradients

                    optimizer.zero_grad() # sets all the model.parameters()'s tensors' gradients to zero
                    loss.backward() # computes the gradients to the model's parameters
                    optimizer.step() # applies gradients

                    bar.set_postfix(mse=float(loss)) # **kwargs

            y_pred = self.model(self.x_val)
            mse = loss_fn(y_pred, self.y_val) # mse computed with this epochs result and y_val
            history.append(mse) # added mse to history
            if (mse < self.best_mse):
                self.mse = mse
                self.best_model = self.model.state_dict() # storing the best model so far

        plt.plot([x.item() for x in history])
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.savefig("Linear Predictor/FFN/FFN1.png")  # savefig() is called before show(), show() creates new plot
        plt.show()

        torch.save(self.best_model, 'Linear Predictor/FFN/FFN1.pth')


# torch.save(model.state_dict(), 'save/to/path/model.pth')
# model = MyModelDefinition(args)
# model.load_state_dict(torch.load('load/from/path/model.pth'))


T = Train("Linear Predictor/FFN/train.csv")
T.read_file()
T.split_data(0.5, 1)

print(torch.load('Linear Predictor/FFN/FFN1.pth'))

model = nn.Sequential(
    nn.Linear(1, 1)
)

T.make_nn(model)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(T.model.parameters(), lr = 1)
num_epochs = 4
batch_size = 10

print(T.model(torch.tensor([5], dtype=torch.float32)))

T.grad_descent(num_epochs, batch_size, loss_fn, optimizer)

print(T.best_model)





