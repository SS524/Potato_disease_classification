import tensorflow as tf
from tensorflow import keras
from utils import load_obj
import os
import numpy as np
from tensorflow.keras.models import load_model as tfk__load_model

class PredictionPipeline:
    def __init__(self,img_arr):
        self.img_arr = img_arr

    def predict(self):
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        img_array = tf.keras.preprocessing.image.img_to_array(self.img_arr)
        img_array = tf.expand_dims(img_array, 0)
        model = tfk__load_model(os.path.join('artifacts','model.h5'))
        predictions = model.predict(img_array)
        print(predictions)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence
