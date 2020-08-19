from ps_model.predict import make_prediction
from ps_model.config import config
from ps_model.processing.data_management import load_data


def test_single_output():
	# Given
	_data = load_data(config.X_TEST_DATA_PATH)
	single_test = _data[0:1]

	# When
	subject = make_prediction(single_test)

	# Then
	assert len(subject) is not None
	assert subject == 6




def test_multiple_output():
	# Given
	_data = load_data(config.X_TEST_DATA_PATH)
	data_length = len(_data)

	# When
	subject = make_prediction(_data)

	# Then
	assert subject is not None
	assert len(subject) == data_length

