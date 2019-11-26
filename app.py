import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.image as mpimg
from imageai.Prediction import ImagePrediction
import os

# Get path
image_path = 'images/'
model_path = 'model/'
model = 'model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'


# Actual Prediction on image using Resnet
prediction = ImagePrediction()
st.cache(prediction.setModelTypeAsResNet())

# model = 'model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

# Set Title
st.title("Image Recognition")

# Set Sidebar Options
st.sidebar.title('About')
st.sidebar.info('Choose an image to test the ResNet50 classifier.')
st.sidebar.title("Predict New Images")
onlyfiles = [f for f in os.listdir(image_path)]

# Select image path from options
imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)

# Read image metadata
img = mpimg.imread(os.path.join(image_path, imageselect))

# Plot on streamlift
st.image(img, caption="Let's predict the image!", use_column_width=True)

if st.sidebar.button('Predict Image'):
    
#     model = 'model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    st.cache(prediction.setModelPath(model))
    st.cache(prediction.loadModel())
    
    # Get predictions
    predictions, probabilities = prediction.predictImage(os.path.join(image_path, imageselect), result_count=5)

    # Print out predictions
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        st.write(eachPrediction , " : " , eachProbability)
