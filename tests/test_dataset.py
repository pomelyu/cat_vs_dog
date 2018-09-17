from torchvision.utils import make_grid, save_image

import context # pylint: disable=unused-import
from dataset import dataloader

class Opt():
    train_data_path = "data/train_mini"
    batch_size = 16

opt = Opt()

train_dataloader = dataloader.create_train_dataloader(opt.train_data_path, opt.batch_size)
valid_dataloader = dataloader.create_valid_dataloader(opt.train_data_path, opt.batch_size)

train_iter = iter(train_dataloader)
valid_iter = iter(valid_dataloader)

train_images, train_labels = next(train_iter)
train_grid = make_grid(train_images, 4)
save_image(train_grid, "out/train_grid.jpg")
print("train_labels:", train_labels.numpy())

valid_images, valid_labels = next(valid_iter)
valid_grid = make_grid(valid_images, 4)
save_image(valid_grid, "out/valid_grid.jpg")
print("valid_labels:", valid_labels.numpy())
