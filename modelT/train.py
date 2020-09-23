import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import model

def load_cifar10(batch=128):
    train_loader = DataLoader(
        datasets.CIFAR10('./data',train=True,download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5,0.5,0.5],
                [0.5,0.5,0.5])
            ])),
        batch_size=batch,
        shuffle=True)

    test_loader = DataLoader(
        datasets.CIFAR10('./data',train=False,download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5,0.5,0.5],
                [0.5,0.5,0.5])
            ])),
        batch_size=batch,
        shuffle=True)

    return {'train': train_loader, 'test': test_loader}

if __name__ == '__main__':
    epoch = 100

    loader = load_cifar10()
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    net: model = model()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

    train_loss = []
    train_acc = []
    test_acc = []

    for e in range(epoch):
        net.train()
        loss = None
        for i, (images, labels) in enumerate(loader['train']):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Training log: {} epoch ({} / 50000 train. data). Loss: {}'.format(e + 1, (i + 1) * 128, loss.item()))

        train_loss.append(loss.item())
        net.eval()
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader['train'])):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        acc = float(correct / 50000)
        train_acc.append(acc)

        correct = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader['test'])):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        acc = float(correct / 10000)
        test_acc.append(acc)

    plt.plot(range(1, epoch+1), train_loss)
    plt.title('Training Loss [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('result/cifar10_loss.png')
    plt.close()

    plt.plot(range(1, epoch + 1), train_acc, label='train_acc')
    plt.plot(range(1, epoch + 1), test_acc, label='test_acc')
    plt.title('Accuracies [CIFAR10]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('result/cifar10_acc.png')
    plt.close()
