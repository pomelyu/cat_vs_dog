from torch.utils.data import DataLoader
from .dataset import DogCatData

def create_train_dataloader(path, batch_size):
    train_dataset = DogCatData(path, usage="train")
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def create_valid_dataloader(path, batch_size):
    valid_dataset = DogCatData(path, usage="valid")
    return DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

def create_test_dataloader(path, batch_size):
    test_dataset = DogCatData(path, usage="test")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
