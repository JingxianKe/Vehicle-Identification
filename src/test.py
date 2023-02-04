import torch.optim as optim
import car_config as config
import numpy as np
import argparse
import pickle
import torch
import time
import os

from torch import nn
from tqdm.auto import tqdm
from model import build_model
from build_dataset import train_set, vaild_set, test_set, get_data_loaders
from torchvision import models

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    "-s", "--start-epoch", type=int, default=0,
    help="epoch to restart training at")
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

# load the label encoder
le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())


def rank5_accuracy(preds, labels):
    # initialize the rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0

    # loop over the predictions and ground-truth labels
    for (p, gt) in zip(preds, labels):
        # sort the probabilities by their index in descending
        # order so that the more confident guesses are at the
        # front of the list
        p = np.argsort(p)[::-1]

        # check if the ground-truth label is in the top-5
        # predictions
        if gt in p[:5]:
            rank5 += 1

        # check to see if the ground-truth is the #1 prediction
        if gt == p[0]:
            rank1 += 1

    # compute the final rank-1 and rank-5 accuracies
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))
    # return a tuple of the rank-1 and rank-5 accuracies
    return (rank1, rank5)

def test(model, testloader):
    model.eval()
    preds = []
    labels = []
    counter = 0

    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        counter += 1

        image, label = data
        image = image.to(device)
        label = label.to(device)
        # Forward pass.
        outputs = model(image)
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy()
        label = label.cpu().numpy()
        # update the predictions and targets lists, respectively
        preds = np.append(preds, preds)
        labels = np.append(labels, label)


    # compute the rank-1 and rank-5 accuracies
    (rank1, rank5) = rank5_accuracy(preds, labels)
    print(counter)
    print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
    print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test = train_set(), vaild_set(), test_set()
    print(f"[INFO]: Number of test images: {len(dataset_test)}")
    # Load the training and validation data loaders.
    train_loader, _, test_loader = get_data_loaders(dataset_train, dataset_valid, dataset_test)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")

    # Load the model.
    model = build_model(
        pretrained=False,
        fine_tune=False,
        num_classes=len(train_set())
    )
    checkpoint = torch.load('./output/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    test(model, train_loader)


