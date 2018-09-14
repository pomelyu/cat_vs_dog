import torch
import numpy as np

from models import ResNet34
from dataset import dataloader
from utils import Visiualizer
from torchnet import meter

class Option():
    def __init__(self):
        self.batch_size = 32
        self.train_data_path = "data/train_mini"
        self.test_data_path = "data/test_mini"
        self.lr = 0.0001
        self.max_epoch = 5
        self.env = "default"
        self.load_model_path = None
        self.use_gpu = True

def train(model, train_dataloader, valid_dataloader, opt):
    vis = Visiualizer(opt.env)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()

        # Train
        model.train()
        
        for _, (data, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()

            prediction = model(data)
            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()

            loss_meter.add(loss.detach().item())
            confusion_matrix.add(prediction.detach(), label)
            vis.plot("loss", loss_meter.value()[0])

        # Save model
        model.save()

        # Valid
        valid_confusion_matrix, valid_acc = valid(model, valid_dataloader)

        vis.plot("val_acc", valid_acc)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss:.3f}, train_cm:{train_cm}, valid_acc:{valid_acc:.2f}".format(
            epoch=epoch,
            lr=opt.lr,
            loss=loss_meter.value()[0],
            train_cm=str(confusion_matrix.value()),
            valid_acc=valid_acc
        ))


def valid(model, data_loader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for _, (data, label) in enumerate(data_loader):
        if opt.use_gpu:
            data = data.cuda()
            label = label.cuda()

        prediction = model(data)

        confusion_matrix.add(prediction.detach().squeeze(), label.long())

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / cm_value.sum()

    return confusion_matrix, accuracy


if __name__ == "__main__":
    opt = Option()

    train_dataloader = dataloader.create_train_dataloader(opt.train_data_path, opt.batch_size)
    valid_dataloader = dataloader.create_valid_dataloader(opt.train_data_path, opt.batch_size)

    model = ResNet34()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = model.cuda()

    train(model, train_dataloader, valid_dataloader, opt)
