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

parser = argparse.ArgumentParser(description="Test CNN")
parser.add_argument("-m", "--model", type=str, help="path to model state dict")
parser.add_argument("-te", "--test", default="data/test.csv", type=str,
                    help="name of trainining-data file (default: data/test.csv)")
parser.add_argument("--out-dir", default="predictions/", type=str,
                    help="directory of prediction file (default: predictions/)")
parser.add_argument("-o", "--out-name", default=None, type=str,
                    help="Name for predictions file (default: None)")
parser.add_argument("--num-workers", default=4, type=int,
                    help="Num-workers for dataloader (default: 4)")
parser.add_argument("--batch-size", default=128, type=int,
                    help="mini-batch size (default: 128)")
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


def test():
    model.eval()
    predictions = []
    for batch in test_loader:
        inputs = Variable(batch["sound"], volatile=True)
        if args.cuda:
            inputs = inputs.cuda()
        filenames = batch["filename"]

        output = model(inputs)
        _, predicted = torch.max(output.data, 1)

        for prediction, filename, input in zip(predicted, filenames, inputs):
            if input.equal(Variable(torch.zeros(input.size())).cuda()):
                predictions.append((filename.rsplit("/")[-1], "silence"))
            else:
                predictions.append((filename.rsplit("/")[-1], id2name[prediction]))

    return predictions


def predictions_to_file(path, predictions):
    with open(path, "w") as f:
        f.write("fname,label\n")
        for prediction, filename in predictions:
            f.write("{},{}\n".format(prediction, filename))

            
if args.out_name is None:
    args.out_name = args.model.split("/")[-1]
    args.out_name = args.out_name.split("_")[0]
 
predictions = test()
predictions_to_file(os.path.join(args.out_dir, args.out_name + ".csv"), predictions)
