from data import get_dataset
from models.alexnet_binary import AlexNetOWT_BN_loss
from preprocess import get_transform
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import accuracy, AverageMeter

id = ''

device = 'cpu'
dataset = 'imagenet'

batch_size = 32

workers = 1

lr = 0.001

EPOCHS = 10

args_distrloss = 0.5

model = AlexNetOWT_BN_loss().to(device)
model.load_state_dict(torch.load('alexnetloss.pt', map_location=torch.device('cpu')))

default_transform = {
    'train': get_transform(dataset,
                           input_size=224, augment=True),
    'eval': get_transform(dataset,
                          input_size=224, augment=False)
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
    tot_tr_distr_loss1 = []
    tot_tr_distr_loss2 = []

    tot_val_losses = []
    tot_val_top1 = []
    tot_val_top5 = []
    tot_val_distr_loss1 = []
    tot_val_distr_loss2 = []

    best_loss = 100

    for epoch in range(EPOCHS):
        if epoch == 5:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr * .1)
        print(epoch)
        tr_losses = AverageMeter()
        tr_top1 = AverageMeter()
        tr_top5 = AverageMeter()
        tr_distr_loss1 = AverageMeter()
        tr_distr_loss2 = AverageMeter()

        print(len(train_loader))
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            out, distr_loss1, distr_loss2 = model(inputs)
            distr_loss1 = distr_loss1.mean()
            distr_loss2 = distr_loss2.mean()
            loss = criterion(out, target)

            loss = loss + (distr_loss1 + distr_loss2) * args_distrloss

            loss.backward()

            optimizer.step()

            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            tr_losses.update(loss.item(), inputs.size(0))
            tr_top1.update(prec1.item(), inputs.size(0))
            tr_top5.update(prec5.item(), inputs.size(0))
            tr_distr_loss1.update(distr_loss1.item(), inputs.size(0))
            tr_distr_loss2.update(distr_loss2.item(), inputs.size(0))

        tot_tr_losses.append(tr_losses.avg)
        tot_tr_top1.append(tr_top1.avg)
        tot_tr_top5.append(tr_top5.avg)
        tot_tr_distr_loss1.append(tr_distr_loss1.avg)
        tot_tr_distr_loss2.append(tr_distr_loss2.avg)

        print(tr_losses.avg)
        print(tr_top1.avg)
        print(tr_top5.avg)
        print(tr_distr_loss1.avg)
        print(tr_distr_loss2.avg)

        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
        val_distr_loss1 = AverageMeter()
        val_distr_loss2 = AverageMeter()
        model.eval()
        for i, (inputs, target) in enumerate(val_loader):
            with torch.no_grad():
                inputs = inputs.to(device)
                target = target.to(device)
                out, distr_loss1, distr_loss2 = model(inputs)
                distr_loss1 = distr_loss1.mean()
                distr_loss2 = distr_loss2.mean()
                loss = criterion(out, target)

                loss = loss + (distr_loss1 + distr_loss2) * args_distrloss

                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                val_losses.update(loss.item(), inputs.size(0))
                val_top1.update(prec1.item(), inputs.size(0))
                val_top5.update(prec5.item(), inputs.size(0))
                val_distr_loss1.update(distr_loss1.item(), inputs.size(0))
                val_distr_loss2.update(distr_loss2.item(), inputs.size(0))

        if val_losses.avg < best_loss:
            torch.save(model.state_dict(), "alexnetloss.pt")

        tot_val_losses.append(val_losses.avg)
        tot_val_top1.append(val_top1.avg)
        tot_val_top5.append(val_top5.avg)
        tot_val_distr_loss1.append(val_distr_loss1.avg)
        tot_val_distr_loss2.append(val_distr_loss2.avg)
        print(val_losses.avg)
        print(val_top1.avg)
        print(val_top5.avg)
        print(val_distr_loss1.avg)
        print(val_distr_loss2.avg)

    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_losses, label='Validation Loss')
    plt.plot(tot_tr_losses, label='Training Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.savefig('lossloss.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_top1, label='Validation Top1 Acc')
    plt.plot(tot_tr_top1, label='Training Top1 Acc')
    plt.legend()
    plt.title('Top1 Acc vs Epoch')
    plt.savefig('top1loss.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_top5, label='Validation Top5 Acc')
    plt.plot(tot_tr_top5, label='Training Top5 Acc')
    plt.legend()
    plt.title('Top5 Acc vs Epoch')
    plt.savefig('top5loss.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_distr_loss1, label='Validation Dist1 Loss')
    plt.plot(tot_tr_distr_loss1, label='Training Dist1 Loss')
    plt.legend()
    plt.title('DistLoss1 vs Epoch')
    plt.savefig('dist1loss.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(tot_val_distr_loss2, label='Validation Dist2 Loss')
    plt.plot(tot_tr_distr_loss2, label='Training Dist2 Loss')
    plt.legend()
    plt.title('DistLoss2 vs Epoch')
    plt.savefig('dist2loss.png')