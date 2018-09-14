import torch
import numpy as np

from models import ResNet34
from dataset import dataloader

batch_size = 32
train_data_path = "data/train_mini"
test_data_path = "data/test_mini"
lr = 0.0001
max_epoch = 2

train_dataloader = dataloader.create_train_dataloader(train_data_path, batch_size)
valid_dataloader = dataloader.create_valid_dataloader(train_data_path, batch_size)

model = ResNet34()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(max_epoch):

    model.train()
    train_loss = []
    for _, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        prediction = model(data)
        loss = criterion(prediction, label)

        print("loss:", loss.detach().numpy())
        train_loss.append(loss.detach().numpy())

        loss.backward()

        optimizer.step()

    print("train_loss:", np.mean(train_loss))

    model.eval()
    valid_loss = []
    for _, (data, label) in enumerate(valid_dataloader):
        prediction = model(data)
        loss = criterion(prediction, label)

        valid_loss.append(loss.detach().numpy())

    print("valid_loss:", np.mean(valid_loss))
        
