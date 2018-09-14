import torch
import numpy as np

from models import ResNet34
from dataset import dataloader
from utils import Visiualizer
from torchnet import meter

batch_size = 32
train_data_path = "data/train_mini"
test_data_path = "data/test_mini"
lr = 0.0001
max_epoch = 2
env = "default"

vis = Visiualizer(env)

train_dataloader = dataloader.create_train_dataloader(train_data_path, batch_size)
valid_dataloader = dataloader.create_valid_dataloader(train_data_path, batch_size)

model = ResNet34()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_meter = meter.AverageValueMeter()
confusion_matrix = meter.ConfusionMeter(2)
previous_loss = 1e100

for epoch in range(max_epoch):

    model.train()
    for _, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        prediction = model(data)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        loss_meter.add(loss.detach()[0].item())
        confusion_matrix.add(prediction.detach(), label)
        vis.plot("loss", loss_meter.value()[0])

    model.save()

    model.eval()
    valid_loss = []
    for _, (data, label) in enumerate(valid_dataloader):
        prediction = model(data)
        loss = criterion(prediction, label)

        valid_loss.append(loss.detach().numpy())

    print("valid_loss:", np.mean(valid_loss))
        
