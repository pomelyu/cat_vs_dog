import os
import torch
import numpy as np
from tqdm import tqdm
from torchnet import meter

from models import ResNet34
from dataset import dataloader
from utils import Visiualizer
from config import TrainOptions

def train(model, train_dataloader, valid_dataloader, opt):
    vis = Visiualizer(opt.env)

    loss_meter = meter.AverageValueMeter()
    acc_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 10e5
    lr = opt.lr

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        acc_meter.reset()
        confusion_matrix.reset()

        # Train
        model.train()
        
        for ii, (data, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True):
            optimizer.zero_grad()

            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()

            prediction = model(data)
            loss = criterion(prediction, label.float())

            loss.backward()
            optimizer.step()

            loss_meter.add(loss.detach().item())
            confusion_matrix.add(prediction.detach() >= 0.5, label)
            cm_value = confusion_matrix.value()
            accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / cm_value.sum()
            acc_meter.add(accuracy)

            if ii % opt.print_freq == opt.print_freq-1:
                vis.plot("loss", loss_meter.value()[0])

            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()

        # Save model
        model.save()

        # Valid
        valid_confusion_matrix, valid_acc = valid(model, valid_dataloader)

        vis.plot("val_acc", valid_acc)
        train_log = "epoch:{epoch}, lr:{lr}, loss:{loss:.3f}, train_cm:{train_cm}, train_acc:{train_acc:.2f}, valid_acc:{valid_acc:.2f}".format(
            epoch=epoch,
            lr=lr,
            loss=loss_meter.value()[0],
            train_cm=str(confusion_matrix.value()),
            train_acc=acc_meter.value()[0],
            valid_acc=valid_acc
        )
        vis.log(train_log)
        print(train_log)

        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        previous_loss = loss_meter.value()[0]


def valid(model, data_loader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for _, (data, label) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True):
        if opt.use_gpu:
            data = data.cuda()
            label = label.cuda()

        prediction = model(data)

        confusion_matrix.add(prediction.detach().squeeze() >= 0.5, label.long())

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / cm_value.sum()

    return confusion_matrix, accuracy


if __name__ == "__main__":
    opt = TrainOptions().parse()

    train_dataloader = dataloader.create_train_dataloader(opt.train_data_path, opt.batch_size)
    valid_dataloader = dataloader.create_valid_dataloader(opt.train_data_path, opt.batch_size)

    model = ResNet34()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = model.cuda()

    train(model, train_dataloader, valid_dataloader, opt)
