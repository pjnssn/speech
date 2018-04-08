import argparse
import os
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import models
from utils import data, util

parser = argparse.ArgumentParser(description="Train CNN using pseudo-labels")
parser.add_argument("-m", "--model", type=str, help="path to model state dict")
parser.add_argument("-t", "--train", default="data/train.csv", type=str,
                    help="name of trainining-data file (default: data/train.csv)")
parser.add_argument("-v", "--validation", default="data/validation.csv", type=str,
                    help="name of validation-data file (default: data/validation.csv)")
parser.add_argument("-b", "--background", default="data/background.csv", type=str,
                    help="name of background-data file (default: data/background.csv)")
parser.add_argument("-te", "--test", default="data/test.csv", type=str,
                    help="name of test-data file (default: data/test.csv)")
parser.add_argument("--save-dir", default="model/saves", type=str,
                    help="directory of saved model (default: model/saves)")
parser.add_argument("--save-prefix", default=None, type=str,
                    help="Prefix for saved model file (default: None)")
parser.add_argument("--batch-size", default=128, type=int,
                    help="mini-batch size (default: 128)")
parser.add_argument("--pseudo-label-ratio", default=0.33, type=float,
                    help="Ratio of pseudo-labels used per batch (default: 0.33)")
parser.add_argument("--pseudo-label-threshold", default=0.95, type=float,
                    help="Threshold for selecting pseudo-labels (default: 0.95)")
parser.add_argument("--resume-training", action="store_true",
                    help="Resume training instead of resetting weights")
parser.add_argument("--T0", default=10, type=int,
                    help="SGDR first cycle length (default: 10)")
parser.add_argument("--mult", default=2, type=int,
                    help="SGDR mult. value (default: 2)")
parser.add_argument("--epochs", default=150, type=int,
                    help="number of total epochs (default: 150)")
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

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

model = models.CNN()
model.load_state_dict(torch.load(args.model))
if args.cuda:
    model.cuda()

print("Model parameter count:", model.parameter_count)

testset = pd.read_csv(args.test)
test = data.AudioPredictionDataset(testset)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)


def get_pseudo_labels(threshold=0.95):
    model.eval()
    pseudo_uids, pseudo_filepaths, pseudo_labels = [], [], []

    for batch in test_loader:
        inputs = Variable(batch["sound"], volatile=True).cuda()
        filenames = batch["filename"]

        output = model(inputs)

        for prediction, filename in zip(output, filenames):
            prediction = F.softmax(prediction, -1)
            prob, pred = torch.max(prediction.data, -1)
            if prob[0] > threshold:
                pseudo_uids.append("pseudo")
                pseudo_filepaths.append(filename)
                pseudo_labels.append(pred)

    return pseudo_uids, pseudo_filepaths, pseudo_labels


pseudo_uids, pseudo_filepaths, pseudo_labels = get_pseudo_labels(args.pseudo_label_threshold)
pseudoset = pd.DataFrame({"uid":pseudo_uids, "file":pseudo_filepaths, "label":pseudo_labels})

trainset = pd.read_csv(args.train)
valset = pd.read_csv(args.validation)
background_set = pd.read_csv(args.background)

train = data.AudioDataset(trainset, background_set)
validation = data.AudioDataset(valset)
pseudo = data.AudioDataset(pseudoset, background_set)

sampler = data.get_sampler(train, round(len(train) * (1 - args.pseudo_label_ratio)))
pseudo_sampler = data.get_sampler(pseudo, round(len(train) * args.pseudo_label_ratio))

train_batch_size = round(args.batch_size * (1 - args.pseudo_label_ratio))
pseudo_batch_size = round(args.batch_size * args.pseudo_label_ratio)
train_loader = DataLoader(train, batch_size=train_batch_size, drop_last=True, shuffle=False,
                          sampler=sampler, num_workers=args.num_workers)
validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)
pseudo_loader = DataLoader(pseudo, batch_size=pseudo_batch_size, drop_last=True, shuffle=False,
                          sampler=pseudo_sampler, num_workers=args.num_workers)

if not args.resume_training:
    model = models.CNN()
    if args.cuda:
        model.cuda()
model_params = util.group_weight(model)

epochs = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = util.CosineAnnealingLR(optimizer, len(train_loader) * args.T0, mult=args.mult)


def train(epoch, print_interval=100):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for i, (tr_batch, ps_batch) in enumerate(zip(train_loader, pseudo_loader)):
        scheduler.step()

        inputs_tr = Variable(tr_batch["sound"])
        inputs_ps = Variable(ps_batch["sound"])
        labels_tr = Variable(tr_batch["label"])
        labels_ps = Variable(ps_batch["label"])
        if args.cuda:
            inputs_tr = inputs_tr.cuda()
            inputs_ps = inputs_ps.cuda()
            labels_tr = labels_tr.cuda()
            labels_ps = labels_ps.cuda()

        inputs = torch.cat((inputs_tr, inputs_ps), 0)
        labels = torch.cat((labels_tr, labels_ps), 0)

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


if args.save_prefix is not None:
    args.save_prefix = args.save_prefix + "_pseudo"
else:
    args.save_prefix = type(model).__name__ + "_pseudo"

train_logger = util.CSVLogger(args.save_prefix + "_train_log", ["epoch", "iteration", "accuracy", "loss"])
validation_logger = util.CSVLogger(args.save_prefix + "_validation_log", ["epoch", "loss", "accuracy"])

training_start = time.time()
for epoch in range(1, epochs + 1):
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
