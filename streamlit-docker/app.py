import streamlit as st
import time
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json

# docker build
test_image_path = '/opt/streamlit_app/test_image/'
model_path = '/opt/streamlit_app/model/ps_resnet.json'
weight_path = '/opt/streamlit_app/model/ps_resnet_weights.h5'

IMAGE_SIZE = 224

# # local streamlit testing
# test_image_path = 'test_image/'
# model_path = 'model/ps_resnet.json'
# weight_path = 'model/ps_resnet_weights.h5'


def map_prediction(result):

	if result == 0:
		return 'Black Grass'
	elif result == 1:
		return 'Charlock'
	elif result == 2:
		return 'Cleavers'
	elif result == 3:
		return 'Common Chickweed'
	elif result == 4:
		return 'Common Wheat'
	elif result == 5:
		return 'Fat Hen'
	else:
		return 'Loose Silky-bent'


def load_cnn_model(model_path, weight_path):

	json_file = open(model_path, 'r')
	loaded_json_model = json_file.read()
	json_file.close()

	model = model_from_json(loaded_json_model)
	model.load_weights(weight_path)
	return model


def make_prediction(image):

	im = cv2.resize(image,(IMAGE_SIZE, IMAGE_SIZE))
	im = im.reshape(1,IMAGE_SIZE, IMAGE_SIZE,3)

	model = load_cnn_model(model_path, weight_path)
	predictions = model.predict_classes(im)
	proba = model.predict_proba(im)


	return predictions[0], max(proba[0])



def run():

	st.title('Plant Seedling Classification with ResNet50')


	# image source
	option = st.radio('', ['Choose a test image', 'Upload your own image'])



	if option == 'Choose a test image':

		test_images = os.listdir(test_image_path)
		test_image = st.selectbox('Please select a test image:', test_images)

		img = cv2.imread(test_image_path + test_image)
		display_image = Image.open(test_image_path+test_image)
		st.image(display_image, use_column_width = False, format = 'PNG')
		
		pred, proba = make_prediction(img)
		pred_results = map_prediction(pred)

		if st.button('Predict'):
			with st.spinner('Making Prediction now...'):
				time.sleep(3)

			st.success(f"The predicted result is {pred_results} with probability of {round(proba*100)}%")


	else: 
		uploaded_file = st.file_uploader('Upload an image here (.png)', type = 'png')


		if uploaded_file is not None:

			display_img = Image.open(uploaded_file)
			st.image(display_img, format = 'PNG', use_column_width = True)

			img = np.array(Image.open(uploaded_file))
			pred, proba = make_prediction(img)
			pred_results = map_prediction(pred)

			if st.button('Predict'):
				with st.spinner('Making Prediction now...'):
					time.sleep(3)

					st.success(f"The predicted result is {pred_results} with probability of {round(proba*100)}%")




if __name__ == '__main__':

	run()






















