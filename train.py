import torch
from torch import nn
from torch.nn import functional as F

from models.VGG16 import form_model
from data.data import *


def train(model, data_loader, optimizer, lr_scheduler, cost_func, epochs, device='cuda', eval_step=50):
    if not (device == 'cuda' and torch.cuda.is_available()):
        device = 'cpu'
    model = model.to(device)

    train_loader, test_loader = data_loader

    model.train()
    print(f'training start with {device}.')
    for e in range(epochs):
        loss_sum = 0
        img_sum = 0
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            out = model(img)
            loss = cost_func(out, label)

            loss.backward()
            optimizer.step()

            loss_sum += loss
            img_sum += 1

        lr = optimizer.param_groups[0]['lr']

        print(f'epoch: {e+1} | loss: {loss_sum / img_sum} | learning_rate: {lr}')
        if (e + 1) % eval_step == 0:
            acc = evaluate(model, test_loader, device)
            print(f'acc: {acc}%')
        lr_scheduler.step()

    acc = evaluate(model, test_loader, device)
    print('over')
    print(f'accurate: {acc}%')

    acc_str = ''.join(str(acc).split('.'))

    torch.save(model.state_dict(), f'weights/VGG16_cifar10_{acc_str}.pth')


def evaluate(model, test_loader, device, pth_path=None):
    correct = 0
    total = 0
    if pth_path:
        model.load_state_dict(torch.load(pth_path))
    model = model.to(device)
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)
            total += len(label)
            out = model(img)
            out = torch.argmax(out, 1)

            correct += torch.where(out == label, 1, 0).sum().item()
    return round(correct / total * 100, 2)


if __name__ == '__main__':
    model = form_model()
    train_datasets, test_datasets = form_datasets()
    train_loader = form_dataloader(
        train_datasets,
        batch_size=64,
        test=False
    )
    test_loader = form_dataloader(
        test_datasets,
        batch_size=64,
        test=True
    )
    loader = (train_loader, test_loader)
    # train the model
    '''
    cost_function = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5, last_epoch=-1)

    train(
        model,
        loader,
        optim,
        scheduler,
        cost_function,
        epochs=50,
        eval_step=10
    )'''

    pth_path = "weights/VGG16_cifar10_8931.pth"
    acc = evaluate(model, test_loader, 'cuda', pth_path)
    print(f'accuracy: {acc}%')


