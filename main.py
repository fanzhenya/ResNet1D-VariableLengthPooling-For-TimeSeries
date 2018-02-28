import argparse
import os
import time
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models import resnet
from torch.optim.lr_scheduler import ReduceLROnPlateau

from variable_length_pooling import VariableLengthPooling

def to_float_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()

def to_long_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).long()

def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array)


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

def get_onehot(b, n_phones, n_frames):
    #b = np.array([0, 2, 5])
    b2 = np.concatenate((b[1:], [n_frames]))
    o = np.zeros((n_frames, n_phones))
    p = np.zeros(n_frames, dtype=int)
    for idx, (s, e) in enumerate(zip(b, b2)):
        p[s:e] = idx
    o[range(n_frames), p] = 1
    # print(o)
    return o

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, for_conv2d=False):
        self.x = x
        self.y = y
        self.for_conv2d = for_conv2d
        self.total_phonemes = sum([len(xi[1]) for xi in x])
        print("n_utters", self.x.shape[0], "total_phonemes", self.total_phonemes)

    def __getitem__(self, idx):
        """
        return: frames, bounds(onehot), labels
        """
        frames = self.x[idx][0]
        bounds = self.x[idx][1]
        n_phones = len(bounds)
        n_frames = len(frames)
        bounds_onehot = get_onehot(bounds, n_phones, n_frames)
        frames = frames.transpose()
        if self.for_conv2d:
            frames = np.expand_dims(frames, axis=0)
        return to_float_tensor(frames), \
            to_float_tensor(bounds_onehot), \
            to_long_tensor(self.y[idx] if self.y is not None else np.array([-1]))

    def __len__(self):
        return self.x.shape[0]


def get_data_loaders(args, for_conv2d=False):
    print("loading data")

    # args.batch_size = 1

    # xtrain = np.load(args.data_dir + '/dev-features.npy')
    # ytrain = np.load(args.data_dir + '/dev-labels.npy')
    xtrain = np.load(args.data_dir + '/train-features.npy')
    ytrain = np.load(args.data_dir + '/train-labels.npy')
    xdev = np.load(args.data_dir + '/dev-features.npy')
    ydev = np.load(args.data_dir + '/dev-labels.npy')

    print("load complete")
    kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyDataset(xtrain, ytrain, for_conv2d=for_conv2d),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        MyDataset(xdev, ydev, for_conv2d=for_conv2d),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, dev_loader

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal(m.weight.data)
        # m.bias.data.zero_()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.firstrun = True
        self.layers = nn.ModuleList([
            nn.Conv1d(40, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),

            # A
            nn.Conv1d(192, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(192, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(192, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(192, 192, 1, padding=0),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),

            # B
            nn.Conv1d(192, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(192, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(192, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(256, 256, 1, padding=0),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            # C
            # nn.Conv1d(256, 256, 3, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv1d(256, 256, 3, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv1d(256, 256, 3, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            #
            # nn.Conv1d(256, 256, 1, padding=0),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),

            # D
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(512, 512, 1, padding=0),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),

            # E
            nn.Conv1d(512, 512, 5, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 512, 7, padding=3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 512, 9, padding=4),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 1024, 11, padding=5),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),

            # nn.Conv1d(256, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv1d(128, 46, 1, padding=0),
            # nn.BatchNorm1d(46),
            # nn.LeakyReLU(inplace=True),

            nn.Conv1d(1024, 1024, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1024, 1024, 3, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1024, 46, 3, padding=1),
            nn.BatchNorm1d(46),
            nn.LeakyReLU(inplace=True),

            VariableLengthPooling()
        ])

    def forward(self, input, bounds=None, print_firstrun=False):
        h = input
        if self.firstrun:
            print("****************************************")
            print("input: {}".format(h.size()))
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and isinstance(layer, VariableLengthPooling):
                h = layer(h, bounds=bounds)
            else:
                h = layer(h)
            if print_firstrun and self.firstrun:
                print("{}: {}".format(layer, h.size()))
        if self.firstrun:
            print("****************************************")
        self.firstrun = False
        return h



def MyModelResNet2D():
    """
    Conv2D Resnet
    :return:
    """
    from my_resnet import ResNet,  BasicBlock
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=46)

def MyModelResNet1D():
    """
    Conv1D Resnet
    :return:
    """
    from my_resnet1d import ResNet, BasicBlock, Bottleneck
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=46)

def train(epoch, model, optimizer, train_loader, args):
    model.train()

    t0 = time.time()
    for batch_idx, (frames, bounds, labels) in enumerate(train_loader):
        if args.cuda:
            frames, bounds, labels = map(lambda x: x.cuda(), [frames, bounds, labels])
        # data, target = Variable(data), Variable(target)
        frames, bounds, labels = map(lambda x: Variable(x), [frames, bounds, labels])
        optimizer.zero_grad()

        data = frames
        output = model(data, bounds=bounds)

        n_phones = len(labels.squeeze())
        # print("n_phones", n_phones)
        loss = F.cross_entropy(output.squeeze().transpose(0, 1), labels.squeeze(), size_average=False)
        # Weighted loss. Typical utterance has 72 phonemes
        weighted_loss = loss * n_phones / 72.0

        # l2 reg
        #         if args.cuda:
        #             l2_reg = Variable(torch.cuda.FloatTensor(1), requires_grad=True)
        #         else:
        #             l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        #         for W in model.parameters():
        #             l2_reg = l2_reg + W.norm(2)

        #         loss += args.l2_reg * l2_reg

        weighted_loss.backward()
        optimizer.step()
        # average loss per phoneme
        avg_loss = loss / n_phones


        if batch_idx % args.log_interval == 0:

            # if avg_loss.data[0] > 3.0:
            #     pred = output.squeeze().transpose(0, 1).data.max(1, keepdim=True)[1]
            #     gt = labels.squeeze().data.view_as(pred)
            #     print(n_phones, loss.data[0], weighted_loss.data[0])
            #     print("gt  ", gt.view(1, -1), "\npred", pred.view(1, -1))

            print('Train Epoch: {} Batch: {} [{}/{} ({:.2f}%, time:{:.2f}s)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), time.time() - t0,
                avg_loss.data[0]))
            t0 = time.time()


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for frames, bounds, labels in test_loader:
        if args.cuda:
            frames, bounds, labels = map(lambda x: x.cuda(), [frames, bounds, labels])
        frames, bounds, labels = Variable(frames, volatile=True), Variable(bounds), Variable(labels)

        data = frames

        output = model(data, bounds=bounds)
        output = output.squeeze().transpose(0, 1)
        labels = labels.squeeze()
        test_loss += F.cross_entropy(output, labels, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    test_loss /= test_loader.dataset.total_phonemes
    accuracy = correct / test_loader.dataset.total_phonemes
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, test_loader.dataset.total_phonemes,
        100 * accuracy))
    return "{:.4f}%".format(100. * correct / test_loader.dataset.total_phonemes), accuracy


def main(args):
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_data_loaders(args, for_conv2d=False)

    model = MyModelResNet1D()


    # model = MyModel()
    # model.apply(weights_init)

    if args.cuda:
        model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5) #1e-4
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True,
                                  threshold_mode='abs', threshold=0.01, min_lr=1e-6)
    for epoch in range(1, args.epochs + 1):
        print(datetime.now())
        train(epoch, model, optimizer, train_loader, args)
        acc_str, acc = test(model, test_loader, args)
        scheduler.step(acc)
        if not os.path.exists(args.weights_dir):
            os.makedirs(args.weights_dir)
        torch.save(model.state_dict(), "{}/{:03d}_{}.w".format(args.weights_dir, epoch, acc_str))


def predict_batch(model, x, bounds, args):
    if args.cuda:
        model.cuda()
        x = x.cuda()
        bounds = bounds.cuda()
    model.eval()
    output = model(Variable(x, volatile=True), bounds=Variable(bounds))
    output = output.squeeze().transpose(0, 1)
    return output.data.max(1, keepdim=True)[1]


def get_test_data_loaders(args):
    print("loading data")
    # args.batch_size = 1
    xtest = np.load(args.data_dir + '/test-features.npy')

    print("load complete")
    # 'num_workers': 8,
    kwargs = {'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        MyDataset(xtest, None),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return test_loader


def predict(args, csv_fpath, weights_fpath):
    model = MyModelResNet2D()
    model.load_state_dict(torch.load(weights_fpath))
    test_loader = get_test_data_loaders(args)
    with open(csv_fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Label'])
        writer.writeheader()
        cnt = 0
        for batch, (frames, bounds, _) in enumerate(test_loader):
            if batch % args.log_interval == 0:
                print("batch", batch)
            yhat = predict_batch(model, frames, bounds, args)
            for i, y in enumerate(yhat[:]):
                writer.writerow({"Id": cnt + i, "Label": y.cpu()[0]})
            cnt += len(yhat)
    print("done")


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--l2-reg', type=float, default=0.001,
                    help='l2 regularization')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--K', type=int, default=10, metavar='N',
#                     help='window size')
parser.add_argument('--data-dir', type=str, default='./data/',
                    help='data directory')
parser.add_argument('--weights-dir', type=str, default='./weights/',
                    help='data directory')


if __name__ == "__main__":
    print(torch.__version__)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.batch_size = 1
    main(args)
    #predict(args, './e1/submission.csv', './e1/weights/012_82.1546%.w')