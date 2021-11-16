from data import get_dataset
from models.vgg_cifar10 import AlexNetOWT_BN
from preprocess import get_transform
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import accuracy, AverageMeter

id_ = 'base'
dataset='cifar10'

batch_size = 32

workers = 1

lr = 0.001

EPOCHS = 50

model = AlexNetOWT_BN()

default_transform = {
        'train': get_transform(dataset,
                               input_size=32, augment=True),
        'eval': get_transform(dataset,
                              input_size=32, augment=False)
    }
transform = getattr(model, 'input_transform', default_transform)

# define loss function (criterion) and optimizer
criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()


if __name__ == "__main__":
    val_data = get_dataset(dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)



    train_data = get_dataset(dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()

    tot_tr_losses = []
    tot_tr_top1 = []
    tot_tr_top5 = []

    tot_val_losses = []
    tot_val_top1 = []
    tot_val_top5 = []

    best_loss = 100

    for epoch in range(EPOCHS):
        tr_losses = AverageMeter()
        tr_top1 = AverageMeter()
        tr_top5 = AverageMeter()


        for i, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, target)
            loss.backward()

            optimizer.step()

            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            tr_losses.update(loss.item(), inputs.size(0))
            tr_top1.update(prec1.item(), inputs.size(0))
            tr_top5.update(prec5.item(), inputs.size(0))

        tot_tr_losses.append(tr_losses.avg)
        tot_tr_top1.append(tr_top1.avg)
        tot_tr_top5.append(tr_top5.avg)

        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        model.eval()
        for i, (inputs, target) in enumerate(val_loader):
            with torch.no_grad():
                out = model(inputs)

                loss = criterion(out, target)

                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                val_losses.update(loss.item(), inputs.size(0))
                val_top1.update(prec1.item(), inputs.size(0))
                val_top5.update(prec5.item(), inputs.size(0))

        if val_losses.avg <best_loss:
            torch.save(model.state_dict(), "alexnet_binary.pt")

        tot_val_losses.append(val_losses.avg)
        tot_val_top1.append(val_top1.avg)
        tot_val_top5.append(val_top5.avg)

    plt.figure(figsize=(8,8))
    plt.plot(tot_val_losses, label= 'Validation Loss')
    plt.plot(tot_tr_losses, label = 'Training Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.savefig('loss.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_top1, label='Validation Top1 Acc')
    plt.plot(tot_tr_top1, label='Training Top1 Acc')
    plt.legend()
    plt.title('Top1 Acc vs Epoch')
    plt.savefig('top1.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_top5, label='Validation Top5 Acc')
    plt.plot(tot_tr_top5, label='Training Top5 Acc')
    plt.legend()
    plt.title('Top5 Acc vs Epoch')
    plt.savefig('top5.png')