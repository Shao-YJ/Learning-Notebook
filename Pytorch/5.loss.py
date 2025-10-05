import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequent = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.sequent(x)
    
train_set = torchvision.datasets.CIFAR10("/home/shaoyj/code/Learning-Notebook/Pytorch/Data/cifar10", train=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10("/home/shaoyj/code/Learning-Notebook/Pytorch/Data/cifar10", train=False, transform=torchvision.transforms.ToTensor())
train_data = DataLoader(train_set, batch_size=1, shuffle=False)
test_set = DataLoader(test_set, batch_size=1, shuffle=False)

writer = SummaryWriter("dataloader")
mymodel = MyModel()
loss = nn.CrossEntropyLoss()
step = 0
for data in train_data:
    images, labels = data
    output = mymodel(images)
    res_loss = loss(output, labels)
    res_loss.backward()
    print(res_loss)

writer.close()
