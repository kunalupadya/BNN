import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import BinarizeLinear, BinarizeConv2d, BinarizeConv2dQ, BinarizeLinearQ, Distrloss_layer, \
    BinaryActivation

__all__ = ['alexnet_binary']

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl=3

        self.c1 = BinarizeConv2d(3, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b1 = nn.BatchNorm2d(int(64 * self.ratioInfl))
        self.h1 = nn.Hardtanh()

        self.c2 = BinarizeConv2d(int(64 * self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(int(192 * self.ratioInfl))
        self.h2 = nn.Hardtanh()

        self.c3 = BinarizeConv2d(int(192 * self.ratioInfl), int(384 * self.ratioInfl), kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(int(384 * self.ratioInfl))
        self.h3 = nn.Hardtanh()

        self.c4 = BinarizeConv2d(int(384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(int(256 * self.ratioInfl))
        self.h4 = nn.Hardtanh()

        self.c5 = BinarizeConv2d(int(256 * self.ratioInfl), 256, kernel_size=3, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b5 = nn.BatchNorm2d(256)
        self.h5 = nn.Hardtanh()


        self.l1 = BinarizeLinear(256 * 6 * 6, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.h11 = nn.Hardtanh()
        # nn.Dropout(0.5),
        self.l2 = BinarizeLinear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.h21 = nn.Hardtanh()
        # nn.Dropout(0.5),
        self.l3 = BinarizeLinear(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(1000)
        self.ls = nn.LogSoftmax()

        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        # }
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.b1(x)
        x1 = x
        x = self.h1(x)

        x = self.c2(x)
        x = self.m2(x)
        x = self.b2(x)
        x2 = x
        x = self.h2(x)

        x = self.c3(x)
        x = self.b3(x)
        x3 = x
        x = self.h3(x)

        x = self.c4(x)
        x = self.b4(x)
        x4 = x
        x = self.h4(x)

        x = self.c5(x)
        x = self.m5(x)
        x = self.b5(x)
        x5 = x
        x = self.h5(x)


        x = x.view(-1, 256 * 6 * 6)
        x = self.l1(x)
        x = self.bn1(x)
        x6 = x
        x = self.h11(x)
        # nn.Dropout(0.5),
        x = self.l2(x)
        x = self.bn2(x)
        x7 = x
        x = self.h21(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = self.ls(x)

        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(6, 6))
        # plt.subplot(3, 3, 1)
        # _ = plt.hist(x1.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 2)
        # _ = plt.hist(x2.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 3)
        # _ = plt.hist(x3.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 4)
        # _ = plt.hist(x4.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 5)
        # _ = plt.hist(x5.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 6)
        # _ = plt.hist(x6.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 7)
        # _ = plt.hist(x7.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        # plt.show()
        return x

class AlexNetOWT_BN_loss(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN_loss, self).__init__()
        self.ratioInfl=3

        self.channels = [3, int(64 * self.ratioInfl), int(192 * self.ratioInfl), int(384 * self.ratioInfl), int(256 * self.ratioInfl), 256]
        self.neurons = [self.channels[5] * 6 * 6, 4096, 4096, num_classes]

        self.c1 = BinarizeConv2d(3, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b1 = nn.BatchNorm2d(int(64 * self.ratioInfl))
        self.h1 = nn.Hardtanh()

        self.c2 = BinarizeConv2d(int(64 * self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(int(192 * self.ratioInfl))
        self.h2 = nn.Hardtanh()

        self.c3 = BinarizeConv2d(int(192 * self.ratioInfl), int(384 * self.ratioInfl), kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(int(384 * self.ratioInfl))
        self.h3 = nn.Hardtanh()

        self.c4 = BinarizeConv2d(int(384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(int(256 * self.ratioInfl))
        self.h4 = nn.Hardtanh()

        self.c5 = BinarizeConv2d(int(256 * self.ratioInfl), 256, kernel_size=3, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b5 = nn.BatchNorm2d(256)
        self.h5 = nn.Hardtanh()


        self.l1 = BinarizeLinear(256 * 6 * 6, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.h11 = nn.Hardtanh()
        # nn.Dropout(0.5),
        self.l2 = BinarizeLinear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.h21 = nn.Hardtanh()
        # nn.Dropout(0.5),
        self.l3 = BinarizeLinear(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(1000)
        self.ls = nn.LogSoftmax()

        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        # }

        self.distrloss_layers = []
        for i in range(1,6):
            self.distrloss_layers.append(Distrloss_layer(self.channels[i]))
        for i in range(1,3):
            self.distrloss_layers.append(Distrloss_layer(self.neurons[i]))

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        loss = []
        x = self.c1(x)
        x = self.m1(x)
        x = self.b1(x)
        loss.append(self.distrloss_layers[0](x))
        x1 = x
        x = self.h1(x)

        x = self.c2(x)
        x = self.m2(x)
        x = self.b2(x)
        loss.append(self.distrloss_layers[1](x))
        x2 = x
        x = self.h2(x)

        x = self.c3(x)
        x = self.b3(x)
        loss.append(self.distrloss_layers[2](x))
        x3 = x
        x = self.h3(x)

        x = self.c4(x)
        x = self.b4(x)
        loss.append(self.distrloss_layers[3](x))
        x4 = x
        x = self.h4(x)

        x = self.c5(x)
        x = self.m5(x)
        x = self.b5(x)
        loss.append(self.distrloss_layers[4](x))
        x5 = x
        x = self.h5(x)


        x = x.view(-1, 256 * 6 * 6)
        x = self.l1(x)
        x = self.bn1(x)
        loss.append(self.distrloss_layers[5](x))
        x6 = x
        x = self.h11(x)
        # nn.Dropout(0.5),
        x = self.l2(x)
        x = self.bn2(x)
        loss.append(self.distrloss_layers[6](x))
        x7 = x
        x = self.h21(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = self.ls(x)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.subplot(3, 3, 1)
        _ = plt.hist(x1.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")

        plt.subplot(3, 3, 2)
        _ = plt.hist(x2.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")

        plt.subplot(3, 3, 3)
        _ = plt.hist(x3.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")

        plt.subplot(3, 3, 4)
        _ = plt.hist(x4.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")

        plt.subplot(3, 3, 5)
        _ = plt.hist(x5.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")

        plt.subplot(3, 3, 6)
        _ = plt.hist(x6.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")

        plt.subplot(3, 3, 7)
        _ = plt.hist(x7.flatten().detach().numpy(), bins=20)
        plt.title("Activation histogram")
        plt.show()
        distrloss1 = sum([ele[0] for ele in loss]) / len(loss)
        distrloss2 = sum([ele[1] for ele in loss]) / len(loss)
        return x, distrloss1.view(1, 1), distrloss2.view(1, 1)


class AlexNetOWT_BN_all(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN_all, self).__init__()
        self.ratioInfl=3

        self.channels = [3, int(64 * self.ratioInfl), int(192 * self.ratioInfl), int(384 * self.ratioInfl), int(256 * self.ratioInfl), 256]
        self.neurons = [self.channels[5] * 6 * 6, 4096, 4096, num_classes]

        self.c1 = BinarizeConv2dQ(3, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b1 = nn.BatchNorm2d(int(64 * self.ratioInfl))
        self.h1 = BinaryActivation()

        self.c2 = BinarizeConv2dQ(int(64 * self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(int(192 * self.ratioInfl))
        self.h2 = BinaryActivation()

        self.c3 = BinarizeConv2dQ(int(192 * self.ratioInfl), int(384 * self.ratioInfl), kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(int(384 * self.ratioInfl))
        self.h3 = BinaryActivation()

        self.c4 = BinarizeConv2dQ(int(384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(int(256 * self.ratioInfl))
        self.h4 = BinaryActivation()

        self.c5 = BinarizeConv2dQ(int(256 * self.ratioInfl), 256, kernel_size=3, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b5 = nn.BatchNorm2d(256)
        self.h5 = BinaryActivation()


        self.l1 = BinarizeLinearQ(256 * 6 * 6, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.h11 = BinaryActivation()
        # nn.Dropout(0.5),
        self.l2 = BinarizeLinearQ(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.h21 = BinaryActivation()
        # nn.Dropout(0.5),
        self.l3 = BinarizeLinearQ(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(1000)
        self.ls = nn.LogSoftmax()

        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        # }

        self.distrloss_layers = []
        for i in range(1,6):
            self.distrloss_layers.append(Distrloss_layer(self.channels[i]))
        for i in range(1,3):
            self.distrloss_layers.append(Distrloss_layer(self.neurons[i]))

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        loss = []
        x = self.c1(x)
        x = self.m1(x)
        x = self.b1(x)
        loss.append(self.distrloss_layers[0](x))
        x1 = x
        x = self.h1(x)

        x = self.c2(x)
        x = self.m2(x)
        x = self.b2(x)
        loss.append(self.distrloss_layers[1](x))
        x2 = x
        x = self.h2(x)

        x = self.c3(x)
        x = self.b3(x)
        loss.append(self.distrloss_layers[2](x))
        x3 = x
        x = self.h3(x)

        x = self.c4(x)
        x = self.b4(x)
        loss.append(self.distrloss_layers[3](x))
        x4 = x
        x = self.h4(x)

        x = self.c5(x)
        x = self.m5(x)
        x = self.b5(x)
        loss.append(self.distrloss_layers[4](x))
        x5 = x
        x = self.h5(x)


        x = x.view(-1, 256 * 6 * 6)
        x = self.l1(x)
        x = self.bn1(x)
        loss.append(self.distrloss_layers[5](x))
        x6 = x
        x = self.h11(x)
        # nn.Dropout(0.5),
        x = self.l2(x)
        x = self.bn2(x)
        loss.append(self.distrloss_layers[6](x))
        x7 = x
        x = self.h21(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = self.ls(x)

        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(6, 6))
        # plt.subplot(3, 3, 1)
        # _ = plt.hist(x1.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 2)
        # _ = plt.hist(x2.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 3)
        # _ = plt.hist(x3.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 4)
        # _ = plt.hist(x4.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 5)
        # _ = plt.hist(x5.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 6)
        # _ = plt.hist(x6.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 7)
        # _ = plt.hist(x7.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        # plt.show()
        distrloss1 = sum([ele[0] for ele in loss]) / len(loss)
        distrloss2 = sum([ele[1] for ele in loss]) / len(loss)
        return x, distrloss1.view(1, 1), distrloss2.view(1, 1)

class AlexNetBNQ(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetBNQ, self).__init__()
        self.ratioInfl = 3
        self.c1 = BinarizeConv2dQ(3, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b1 = nn.BatchNorm2d(int(64 * self.ratioInfl))
        self.h1 = nn.Hardtanh()

        self.c2 = BinarizeConv2dQ(int(64 * self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(int(192 * self.ratioInfl))
        self.h2 = nn.Hardtanh()

        self.c3 = BinarizeConv2dQ(int(192 * self.ratioInfl), int(384 * self.ratioInfl), kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(int(384 * self.ratioInfl))
        self.h3 = nn.Hardtanh()

        self.c4 = BinarizeConv2dQ(int(384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(int(256 * self.ratioInfl))
        self.h4 = nn.Hardtanh()

        self.c5 = BinarizeConv2dQ(int(256 * self.ratioInfl), 256, kernel_size=3, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b5 = nn.BatchNorm2d(256)
        self.h5 = nn.Hardtanh()


        self.l1 = BinarizeLinearQ(256 * 6 * 6, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.h11 = nn.Hardtanh()
        # nn.Dropout(0.5),
        self.l2 = BinarizeLinearQ(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.h21 = nn.Hardtanh()
        # nn.Dropout(0.5),
        self.l3 = BinarizeLinearQ(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(1000)
        self.ls = nn.LogSoftmax()

        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        # }
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.b1(x)
        x = self.h1(x)

        x = self.c2(x)
        x = self.m2(x)
        x = self.b2(x)
        x = self.h2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.h3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.h4(x)

        x = self.c5(x)
        x = self.m5(x)
        x = self.b5(x)
        x = self.h5(x)


        x = x.view(-1, 256 * 6 * 6)
        x = self.l1(x)
        x = self.bn1(x)
        x = self.h11(x)
        # nn.Dropout(0.5),
        x = self.l2(x)
        x = self.bn2(x)
        x = self.h21(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = self.ls(x)
        return x

class AlexNetBNQA(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetBNQA, self).__init__()
        self.ratioInfl = 3
        self.c1 = BinarizeConv2dQ(3, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b1 = nn.BatchNorm2d(int(64 * self.ratioInfl))
        self.h1 = BinaryActivation()

        self.c2 = BinarizeConv2dQ(int(64 * self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(int(192 * self.ratioInfl))
        self.h2 = BinaryActivation()

        self.c3 = BinarizeConv2dQ(int(192 * self.ratioInfl), int(384 * self.ratioInfl), kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(int(384 * self.ratioInfl))
        self.h3 = BinaryActivation()

        self.c4 = BinarizeConv2dQ(int(384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(int(256 * self.ratioInfl))
        self.h4 = BinaryActivation()

        self.c5 = BinarizeConv2dQ(int(256 * self.ratioInfl), 256, kernel_size=3, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b5 = nn.BatchNorm2d(256)
        self.h5 = BinaryActivation()


        self.l1 = BinarizeLinearQ(256 * 6 * 6, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.h11 = BinaryActivation()
        # nn.Dropout(0.5),
        self.l2 = BinarizeLinearQ(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.h21 = BinaryActivation()
        # nn.Dropout(0.5),
        self.l3 = BinarizeLinearQ(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(1000)
        self.ls = nn.LogSoftmax()

        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        # }
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.b1(x)
        x = self.h1(x)

        x = self.c2(x)
        x = self.m2(x)
        x = self.b2(x)
        x = self.h2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.h3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.h4(x)

        x = self.c5(x)
        x = self.m5(x)
        x = self.b5(x)
        x = self.h5(x)


        x = x.view(-1, 256 * 6 * 6)
        x = self.l1(x)
        x = self.bn1(x)
        x = self.h11(x)
        # nn.Dropout(0.5),
        x = self.l2(x)
        x = self.bn2(x)
        x = self.h21(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = self.ls(x)
        return x

class AlexNetOWT_approx(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_approx, self).__init__()
        self.ratioInfl=3

        self.c1 = BinarizeConv2d(3, int(64 * self.ratioInfl), kernel_size=11, stride=4, padding=2)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b1 = nn.BatchNorm2d(int(64 * self.ratioInfl))
        self.h1 = BinaryActivation()

        self.c2 = BinarizeConv2d(int(64 * self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.BatchNorm2d(int(192 * self.ratioInfl))
        self.h2 = BinaryActivation()

        self.c3 = BinarizeConv2d(int(192 * self.ratioInfl), int(384 * self.ratioInfl), kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(int(384 * self.ratioInfl))
        self.h3 = BinaryActivation()

        self.c4 = BinarizeConv2d(int(384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(int(256 * self.ratioInfl))
        self.h4 = BinaryActivation()

        self.c5 = BinarizeConv2d(int(256 * self.ratioInfl), 256, kernel_size=3, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b5 = nn.BatchNorm2d(256)
        self.h5 = BinaryActivation()


        self.l1 = BinarizeLinear(256 * 6 * 6, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.h11 = BinaryActivation()
        # nn.Dropout(0.5),
        self.l2 = BinarizeLinear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.h21 = BinaryActivation()
        # nn.Dropout(0.5),
        self.l3 = BinarizeLinear(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(1000)
        self.ls = nn.LogSoftmax()

        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        # }
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.b1(x)
        x1 = x
        x = self.h1(x)

        x = self.c2(x)
        x = self.m2(x)
        x = self.b2(x)
        x2 = x
        x = self.h2(x)

        x = self.c3(x)
        x = self.b3(x)
        x3 = x
        x = self.h3(x)

        x = self.c4(x)
        x = self.b4(x)
        x4 = x
        x = self.h4(x)

        x = self.c5(x)
        x = self.m5(x)
        x = self.b5(x)
        x5 = x
        x = self.h5(x)


        x = x.view(-1, 256 * 6 * 6)
        x = self.l1(x)
        x = self.bn1(x)
        x6 = x
        x = self.h11(x)
        # nn.Dropout(0.5),
        x = self.l2(x)
        x = self.bn2(x)
        x7 = x
        x = self.h21(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = self.ls(x)

        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(6, 6))
        # plt.subplot(3, 3, 1)
        # _ = plt.hist(x1.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 2)
        # _ = plt.hist(x2.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 3)
        # _ = plt.hist(x3.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 4)
        # _ = plt.hist(x4.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 5)
        # _ = plt.hist(x5.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 6)
        # _ = plt.hist(x6.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        #
        # plt.subplot(3, 3, 7)
        # _ = plt.hist(x7.flatten().detach().numpy(), bins=20)
        # plt.title("Activation histogram")
        # plt.show()
        return x


def alexnet_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
