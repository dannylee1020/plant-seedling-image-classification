import pathlib
import ps_model

PACKAGE_DIR = pathlib.Path(ps_model.__file__).resolve().parent

TRAINED_MODEL_PATH = PACKAGE_DIR/'trained_model'

# MODEL_NAME = 'plant_seedling_model'
# MODEL_PATH = f"/Users/dhyungseoklee/Projects/ML_Pipelines/plant-seedling-image-detection/pre_trained_model/ps_model/trained_model/{MODEL_NAME}"

MODEL_NAME = 'ps_resnet.json'
MODEL_PATH = TRAINED_MODEL_PATH/MODEL_NAME

WEIGHT_NAME = 'ps_resnet_weights.h5'
WEIGHT_PATH = f"/Users/dhyungseoklee/Projects/ML_Pipelines/plant-seedling-image-detection/pre_trained_model/ps_model/trained_model/{WEIGHT_NAME}"

ENCODER_PATH = TRAINED_MODEL_PATH/'enc.pkl'
DATA_PATH = PACKAGE_DIR/'datasets' 


# X_TRAIN_NAME = 'X_train_resized.npy'
X_TEST_NAME = 'X_test_resized.npy'

Y_TRAIN_NAME = 'y_train.pkl'
Y_TEST_NAME = 'y_test.pkl'

X_TEST_DATA_PATH = DATA_PATH/X_TEST_NAME
Y_TEST_DATA_PATH = DATA_PATH/Y_TEST_NAME

IMAGE_SIZE = 224
