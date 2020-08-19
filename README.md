# Image Classification with ResNet50, Streamlit and Docker


## Project Overview
 Using ResNet50 to make prediction on plant seedling images, creating an app with Streamlit, Docker and deploying it to Heroku. Either choose a test image or upload an image of different types of plant seedling to perform image classification prediction. 

## Model Overview
Data is from [Kaggle](https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset?). Only using first seven target classes and limiting training images to 250 for each class to make training session faster. Therefore, the model may suffer tradeoff in accuracy when it comes to model prediction. Here I am using pre-trained ResNet 50 model and separately adding fully connected layers on the top to make train and make predictions. Transfer Learning is especially useful when it comes to deep learning as it is computationally expensive and time consuming to train a deep neural network on images.  


## Run locally with Docker
From the `streamlit-docker` dir:

        docker build -t dannylee1020/ps_project .
        docker run -p 8501:8501 dannylee1020/ps_project:latest

Then visit [localhost:8501](https://localhost:8501) for streamlit app. Make sure to use local testing commands in Dockerfile when running. Commands for Heroku deployment is default. 


## Testing Model
Using pytest for testing model. simply run `pytest test_predict.py` in tests directory