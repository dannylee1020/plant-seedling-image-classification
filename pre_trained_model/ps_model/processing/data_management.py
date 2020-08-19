from ps_model.config import config

import numpy as np
from tensorflow.keras.models import model_from_json



def load_data(filepath):
	
	data = np.load(filepath, 'r')
	return data



def load_cnn_model(filepath):

	json_file = open(filepath, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	return model


	
	








