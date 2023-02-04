# import the necessary packages
from os import path

# define the base path to the cars dataset
BASE_PATH = "/kaggle/input/cars196"

# based on the base path, derive the images path and meta file path
IMAGES_PATH = path.sep.join([BASE_PATH, "car_ims"])
LABELS_PATH = "./complete_dataset.csv"

# define the path to the label encoder
LABEL_ENCODER_PATH = "./output/le.cpickle"

# define the percentage of validation and testing images relative
# to the number of training images
NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size
BATCH_SIZE = 32
NUM_DEVICES = 1