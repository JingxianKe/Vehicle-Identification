import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import car_config as config
import logging

from tqdm.auto import tqdm

from model import build_model
from build_dataset import train_set, vaild_set, get_data_loaders
from utils import save_model, save_plots
# due to mxnet seg-fault issue, need to place OpenCV import at the
# top of the file
import cv2
# import the necessary packages
from config import car_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import os

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

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
					filename="training_{}.log".format(args["start_epoch"]),
					filemode="w")

# Validation function.
def validate(model, testloader, criterion, class_names):
	model.eval()
	print('Validation')
	valid_running_loss = 0.0
	valid_running_correct = 0
	counter = 0

	with torch.no_grad():
		for i, data in tqdm(enumerate(testloader), total=len(testloader)):
			counter += 1

			image, labels = data
			image = image.to(device)
			labels = labels.to(device)
			# Forward pass.
			outputs = model(image)
			# Calculate the loss.
			loss = criterion(outputs, labels)
			valid_running_loss += loss.item()
			# Calculate the accuracy.
			_, preds = torch.max(outputs.data, 1)
			valid_running_correct += (preds == labels).sum().item()

	# Loss and accuracy for the complete epoch.
	epoch_loss = valid_running_loss / counter
	epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
	return epoch_loss, epoch_acc


if __name__ == '__main__':
	# Load the training and validation datasets.
	dataset_train, dataset_valid = train_set(), vaild_set()
	print(f"[INFO]: Number of training images: {len(dataset_train)}")
	print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
	# Load the training and validation data loaders.
	train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

	# Learning_parameters.
	lr = args['learning_rate']
	epochs = args['epochs']
	device = ('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Computation device: {device}")
	print(f"Learning rate: {lr}")
	print(f"Epochs to train for: {epochs}\n")

	# Load the model.
	model = build_model(
		pretrained=True,
		fine_tune=True,
		num_classes=len(train_set())
	).to(device)

	# Total parameters and trainable parameters.
	total_params = sum(p.numel() for p in model.parameters())
	print(f"{total_params:,} total parameters.")
	total_trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad)
	print(f"{total_trainable_params:,} training parameters.")

	# Optimizer.
	optimizer = optim.SGD(learning_rate=1e-4, momentum=0.9, wd=0.0005)
	# Loss function.
	criterion = nn.CrossEntropyLoss()

	# Lists to keep track of losses and accuracies.
	train_loss, valid_loss = [], []
	train_acc, valid_acc = [], []
	# Start the training.
	for epoch in range(epochs):
		print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
		train_epoch_loss, train_epoch_acc = train(model, train_loader,
												  optimizer, criterion)
		valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
													 criterion, dataset_classes)
		train_loss.append(train_epoch_loss)
		valid_loss.append(valid_epoch_loss)
		train_acc.append(train_epoch_acc)
		valid_acc.append(valid_epoch_acc)
		print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
		print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
		print('-' * 50)
		time.sleep(2)

	# Save the trained model weights.
	save_model(epochs, model, optimizer, criterion)
	# Save the loss and accuracy plots.
	save_plots(train_acc, valid_acc, train_loss, valid_loss)
	print('TRAINING COMPLETE')