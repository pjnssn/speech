import argparse
import os
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import models
from utils import data, util

parser = argparse.ArgumentParser(description="Train CNN")
parser.add_argument("-t", "--train", default="data/train.csv", type=str,
                    help="name of trainining-data file (default: data/train.csv)")
parser.add_argument("-v", "--validation", default="data/validation.csv", type=str,
                    help="name of validation-data file (default: data/validation.csv)")
parser.add_argument("-b", "--background", default="data/background.csv", type=str,
                    help="name of background-data file (default: data/background.csv)")
parser.add_argument("--save-dir", default="model/saves", type=str,
                    help="directory of saved model (default: model/saves)")
parser.add_argument("--save-prefix", default=None, type=str,
                    help="Prefix for saved model file (default: None)")
parser.add_argument("--batch-size", default=128, type=int,
                    help="mini-batch size (default: 128)")
parser.add_argument("--T0", default=10, type=int,
                    help="SGDR first cycle length (default: 10)")
parser.add_argument("--mult", default=2, type=int,
                    help="SGDR mult. value (default: 2)")
parser.add_argument("--epochs", default=70, type=int,
                    help="number of total epochs (default: 70)")
parser.add_argument("-lr", "--learning-rate", default=0.1, type=float,
                    help="Initial learning rate (default: 0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum for optimizer (default: 0.9)")
parser.add_argument("--weight-decay", default=0.0005, type=float,
                    help="Weight-decay for optimizer (default: 0.0005)")
parser.add_argument("--num-workers", default=4, type=int,
                    help="Num-workers for each dataloader (default: 4)")
parser.add_argument("--disable-cuda", action="store_true",
                    help="Disable CUDA")
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

trainset = pd.read_csv(args.train)
valset = pd.read_csv(args.validation)
background_set = pd.read_csv(args.background)

train = data.AudioDataset(trainset, background_set)
validation = data.AudioDataset(valset)

sampler = data.get_sampler(train)

train_loader = DataLoader(train, batch_size=args.batch_size, drop_last=True, shuffle=False,
                          sampler=sampler, num_workers=args.num_workers)
validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)

model = models.CNN()
if args.cuda:
    model.cuda()
model_params = util.group_weight(model)

print("Model parameter count:", model.parameter_count)

epochs = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = util.CosineAnnealingLR(optimizer, len(train_loader) * args.T0, mult=args.mult)


def train(epoch, print_interval=100):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for i, batch in enumerate(train_loader):
        scheduler.step()

        inputs = Variable(batch["sound"])
        labels = Variable(batch["label"])
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        running_loss += loss.data[0]

        if i % print_interval == print_interval - 1:
            print("Epoch: {} - Iteration: {} - Loss: {}".format(epoch, i + 1, running_loss / print_interval))
            train_logger.log(
                {"epoch": epoch, "iteration": i + 1, "loss": running_loss / print_interval, "accuracy": None})
            running_loss = 0.0

    accuracy = correct / total * 100
    training_loss = running_loss / (len(train_loader) % print_interval)
    return accuracy, training_loss


def evaluate():
    model.eval()
    validation_loss = 0.0
    correct, total = 0, 0
    for i, batch in enumerate(validation_loader):
        inputs = Variable(batch["sound"], volatile=True)
        labels = Variable(batch["label"], volatile=True)
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        output = model(inputs)
        loss = criterion(output, labels)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        validation_loss += loss.data[0]

    accuracy = correct / total * 100
    validation_loss /= i
    return accuracy, validation_loss


if args.save_prefix is None:
    args.save_prefix = type(model).__name__

train_logger = util.CSVLogger(args.save_prefix + "_train_log", ["epoch", "iteration", "accuracy", "loss"])
validation_logger = util.CSVLogger(args.save_prefix + "_validation_log", ["epoch", "loss", "accuracy"])

training_start = time.time()
for epoch in range(epochs):
    epoch += 1
    epoch_start = time.time()

    accuracy, loss = train(epoch)
    validation_accuracy, validation_loss = evaluate()

    end = time.time()
    h, m, s = util.to_hms(epoch_start, end)
    hh, mm, ss = util.to_hms(training_start, end)

    print("Epoch: {}:".format(epoch))
    print("Training accuracy: {}".format(accuracy))
    print("Training loss: {}".format(loss))
    print("Validation accuracy: {}".format(validation_accuracy))
    print("Validation loss: {}".format(validation_loss))
    print("Epoch time: {:.4}h {:.4}m {:.4}s".format(h, m, s))
    print("Total time: {:.4}h {:.4}m {:.4}s\n".format(hh, mm, ss))

    train_logger.log({"epoch": epoch, "iteration": len(train_loader), "accuracy": accuracy, "loss": loss})
    validation_logger.log({"epoch": epoch, "accuracy": validation_accuracy, "loss": validation_loss})

save_path = os.path.join(args.save_dir,
                         args.save_prefix + "_acc_{}_loss_{}.pt".format(validation_accuracy, validation_loss))
torch.save(model.state_dict(), save_path)