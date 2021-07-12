import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
args = parser.parse_args()

class Net(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(Net, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)
        return x

    @classmethod
    def load_model_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model):
        model = model
        package = {
            'state_dict': model.state_dict()
        }
        return package

#class Net(nn.Module):
    #def __init__(self):
        #super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    #def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x
    #@classmethod
    #def load_model_package(cls, package):
        #model = cls()
        #model.load_state_dict(package['state_dict'])
        #return model

    #@staticmethod
    #def serialize(model):
        #model = model
        #package = {
            #'state_dict': model.state_dict()
        #}
        #return package

"""def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=True):
    Construct FGSM adversarial examples on the examples X
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        #delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    optimizer = optim.Adam([delta],lr = 0.1)
    X_adv = X + delta
    for t in range(num_iter):
        delta.data = delta.data*2
        loss = nn.CrossEntropyLoss()(model(X_adv), y)
        #delta.retain_grad()
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        #optimizer.step()
        #X_adv = X + delta
        #delta.data = (delta + alpha*torch.sign(delta.grad)).clamp(-epsilon,epsilon)
        #delta.grad.zero_()
        #delta.data = delta.data.clamp(-epsilon,epsilon)
    #return delta.detach()
"""
def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
                               
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        #print(delta.grad)
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
device = 'cuda:0'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def epoch(loader, model, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for i,(data) in tqdm(enumerate(loader)):
            X , y = data
            X,y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for i,(data) in tqdm(enumerate(loader)):
            X ,y =data
            X,y = X.to(device), y.to(device)
            delta = attack(model, X, y, **kwargs)
            yp = model(X+delta)
            loss = nn.CrossEntropyLoss()(yp,y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    criterion = nn.CrossEntropyLoss()
    
    model = Net()
    if args.continue_from:
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = Net.load_model_package(package)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model = model.to(device)
    for t in range(80):
        model.train()
        for child in model.children():
            for param in child.parameters():
                param.requires_grad =True
        for m in model.modules():
            if isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.Dropout):
                m.train()
        print('entering_loop_{i}'.format(i =t))
        train_err, train_loss = epoch(trainloader, model,optimizer)
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        for m in model.modules():
            if isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.Dropout):
                m.eval()
        adv_err, adv_loss = epoch_adversarial(testloader, model, pgd_linf)
        with torch.no_grad():
            test_err, test_loss = epoch(testloader, model)
            if t%30 == 0 and t>1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"]/10
            print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
            torch.save(Net.serialize(model),"cifar/model_cifar10.pth")

    print("PGD, 40 iter: ", epoch_adversarial(testloader, model, pgd_linf, num_iter=40)[0])
