import argparse
import os
import re
import sys
from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Create CSV files for training and validation data")
parser.add_argument("-d", "--data-dir", default="/home/data/audio/data", type=str,
                    help="location of main data directory (default: /home/data/audio/data)")
parser.add_argument("-t", "--train", default="train.csv", type=str,
                    help="name of trainining-data file (default: train.csv)")
parser.add_argument("-v", "--validation", default="validation.csv", type=str,
                    help="name of validation-data file (default: validation.csv)")
parser.add_argument("-b", "--background", default="background.csv", type=str,
                    help="name of background-data file (default: background.csv)")
parser.add_argument("--validation-split", default=0.1, type=float,
                    help="ratio of data used for validation, [0.0, 1.0) (default: 0.1)")
parser.add_argument("-te", "--test", default="test.csv", type=str,
                    help="name of test-data file (default: test.csv)")

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}


def load_data(data_dir, validation_cut=0.1):
    """
    Validation silence entirely from miss-labeled data (silences.txt)
    Training silence from silences.txt + placeholders for silence generation
    """
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, "train/audio/*/*wav"))

    # Text file containing miss-labeled files which are actually silence
    with open(os.path.join(sys.path[0], "silences.txt")) as f:
        silences = [data_dir + "/train/audio/" + i.strip() for i in f]

    filepaths = []
    labels = []
    uids = []
    background_set = set()
    possible = set(POSSIBLE_LABELS)

    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            fp, label, uid = r.group(0), r.group(2), r.group(3)

        if label == "_background_noise_":
            background_set.add(entry)
            continue
        if entry in silences:
            label = "silence"
        if label not in possible:
            label = 'unknown'

        filepaths.append(fp)
        labels.append(name2id[label])
        uids.append(uid)

    df = pd.DataFrame({"uid": uids, "file": filepaths, "label": labels})

    keyword_count = df.label[(df.label != name2id["unknown"]) & (df.label != name2id["silence"])].count()
    keyword_mean = keyword_count // (len(POSSIBLE_LABELS) - 2)

    users = df.uid.drop_duplicates()
    train_users, val_users = train_test_split(users, test_size=validation_cut)

    trainset = df[df.uid.isin(train_users)]
    valset = df[df.uid.isin(val_users)]

    silence_count = trainset.label.value_counts()[name2id["silence"]]
    silence_to_add = keyword_mean - silence_count

    s_uids, s_filepaths, s_labels = [], [], []
    for i in range(silence_to_add):
        s_uids.append("silence_" + str(i))
        s_filepaths.append("silence")
        s_labels.append(name2id["silence"])

    silence = pd.DataFrame({"uid": s_uids, "file": s_filepaths, "label": s_labels})
    trainset = trainset.append(silence)

    trainset.reset_index(inplace=True, drop=True)
    valset.reset_index(inplace=True, drop=True)

    background_set = pd.DataFrame({"file": list(background_set)})

    print("There are {} train and {} validation samples".format(len(trainset), len(valset)))
    return trainset, valset, background_set


def main(args):
    trainset, valset, background_set = load_data(args.data_dir, validation_cut=args.validation_split)

    test_files = glob(os.path.join(args.data_dir, "test/audio/*wav"))
    testset = pd.DataFrame({"file": test_files})

    trainset.to_csv(os.path.join(sys.path[0], args.train))
    valset.to_csv(os.path.join(sys.path[0], args.validation))
    background_set.to_csv(os.path.join(sys.path[0], args.background))
    testset.to_csv(os.path.join(sys.path[0], args.test))


if __name__ == '__main__':
    main(parser.parse_args())