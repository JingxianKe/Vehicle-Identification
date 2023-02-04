# import the necessary packages
import car_config as config
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = config.BATCH_SIZE * config.NUM_DEVICES
NUM_WORKERS = 4 # Number of parallel processes for data preparation.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
import os

# read the contents of the labels file, then initialize the list of
# image paths and labels
print("[INFO] loading image paths and labels...")
rows = open(config.LABELS_PATH).read()
rows = rows.strip().split("\n")[:]
trainPaths = []
trainLabels = []

# loop over the rows
for row in rows:
	# unpack the row, then update the image paths and labels list
	# (filename, make) = row.split(",")[:2]
	(filename, label) = row.split(",")[1:]
	trainPaths.append(os.sep.join([config.IMAGES_PATH, filename]))
	trainLabels.append("{}".format(label))

# now that we have the total number of images in the dataset that
# can be used for training, compute the number of images that
# should be used for validation and testing
numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)

# our class labels are represented as strings so we need to encode
# them
print("[INFO] encoding labels...")
le = LabelEncoder().fit(trainLabels)
trainLabels = le.transform(trainLabels)

# perform sampling from the training set to construct a a validation
# set
print("[INFO] constructing validation data...")
split = train_test_split(trainPaths, trainLabels, test_size=numVal,
						 stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

# perform stratified sampling from the training set to construct a
# a testing set
print("[INFO] constructing testing data...")
split = train_test_split(trainPaths, trainLabels, test_size=numTest,
						 stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

# write the label encoder to file
print("[INFO] serializing label encoder...")
f = open(config.LABEL_ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()

# Training transforms
def get_train_transform(IMAGE_SIZE):
	train_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(15),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
			)
	])
	return train_transform

# Validation transforms
def get_valid_transform(IMAGE_SIZE):
	valid_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
			)
	])
	return valid_transform

def get_test_transform(IMAGE_SIZE):
	test_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
			)
	])
	return test_transform

train_transform = get_train_transform(IMAGE_SIZE)
valid_transform = get_valid_transform(IMAGE_SIZE)
test_transform = get_test_transform(IMAGE_SIZE)

class train_set(Dataset):
	def __init__(self):
		self.imgs = trainPaths
		self.label = trainLabels

	def __getitem__(self, index):
		fn = self.imgs[index]
		label = self.label[index]
		img = Image.open(fn).convert('RGB')
		img = train_transform(img)

		return img, label

	def __len__(self):
		return len(self.imgs)

class vaild_set(Dataset):
	def __init__(self):
		self.imgs = valPaths
		self.label = valLabels

	def __getitem__(self, index):
		fn = self.imgs[index]
		label = self.label[index]
		img = Image.open(fn).convert('RGB')
		img = valid_transform(img)

		return img, label

	def __len__(self):
		return len(self.imgs)

class test_set(Dataset):
	def __init__(self):
		self.imgs = testPaths
		self.label = testLabels

	def __getitem__(self, index):
		fn = self.imgs[index]
		label = self.label[index]
		img = Image.open(fn).convert('RGB')
		img = test_transform(img)

		return img, label

	def __len__(self):
		return len(self.imgs)

def get_data_loaders(dataset_train, dataset_valid, dataset_test):
	"""
	Prepares the training and validation data loaders.
	:param dataset_train: The training dataset.
	:param dataset_valid: The validation dataset.
	Returns the training and validation data loaders.
	"""
	train_loader = DataLoader(
		dataset_train, batch_size=BATCH_SIZE,
		shuffle=True, num_workers=NUM_WORKERS
	)
	valid_loader = DataLoader(
		dataset_valid, batch_size=BATCH_SIZE,
		shuffle=False, num_workers=NUM_WORKERS
	)
	test_loader = DataLoader(
		dataset_test, batch_size=BATCH_SIZE,
		shuffle=False, num_workers=NUM_WORKERS
	)
	return train_loader, valid_loader, test_loader
