import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import model

# say don't repeat your selves

def load_images(batch=32):
    ratio=0.8 # 8 training :2 test
    size = (224, 224)
    train_loader = DataLoader(
        datasets.ImageFolder('./data/downloads',transform=transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5,0.5,0.5],
                [0.5,0.5,0.5])
            ])),
        batch_size=batch,
        shuffle=True)

    print(len(train_loader.dataset))

    n=len(train_loader.dataset)
    a=(int)(n*ratio)

    train_loader, test_loader = torch.utils.data.random_split(train_loader.dataset, [a, n-a])
   
    train_loader = DataLoader(train_loader)
    test_loader = DataLoader(test_loader)

    print(test_loader.dataset)
    print(train_loader.dataset)
    print(len(train_loader))
    print(len(test_loader))
    return {'train': train_loader,'test':test_loader}

def load_cifar10(batch=32):
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
    batch_size = 100

    loader = load_images(batch=batch_size)
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    net: model = model()
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.00001)

    train_loss = []
    train_acc = []
    test_acc = []

    len_of_epoch = batch_size * len(loader['train'])

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
                print('Training log: {} epoch ({} / {} train. data). Loss: {}'.format(e + 1, (i + 1) * batch_size, len_of_epoch, loss.item()))

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

        acc = float(correct / len(loader['train']))
        print('accuracy of train data:{}'.format(acc))
        train_acc.append(acc)

        correct = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader['test'])):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        acc = float(correct / len(loader['test']))
        print('accuracy of test data:{}'.format(acc))
        test_acc.append(acc)

    plt.plot(range(1, epoch+1), train_loss)
    plt.title('Training Loss [Ford vs Ferrari]')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('result/fvf_loss.png')
    plt.close()

    plt.plot(range(1, epoch + 1), train_acc, label='train_acc')
    plt.plot(range(1, epoch + 1), test_acc, label='test_acc')
    plt.title('Accuracies [Ford vs Ferrari]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('result/fvf_acc.png')
    plt.close()
