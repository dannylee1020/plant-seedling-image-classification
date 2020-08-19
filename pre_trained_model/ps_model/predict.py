
from ps_model.config import config
from ps_model.processing.data_management import load_cnn_model, load_data

import joblib
from sklearn.metrics import accuracy_score, classification_report




def make_prediction(data):

	model = load_cnn_model(config.MODEL_PATH)
	model.load_weights(config.WEIGHT_PATH)
	model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	predictions = model.predict_classes(data)

	return predictions



if __name__ == '__main__':

	test_data = load_data(config.X_TEST_DATA_PATH)

	y_pred = make_prediction(test_data)
	y_true = joblib.load(config.Y_TEST_DATA_PATH)

	# encode target
	enc = joblib.load(config.ENCODER_PATH)
	y_true_enc = enc.transform(y_true)

	acc = accuracy_score(y_true_enc, y_pred)
	cf = classification_report(y_true_enc, y_pred)

	print(f"Model Accuracy: {acc}")
	print()
	print(cf)
	print()
	print(y_pred)








