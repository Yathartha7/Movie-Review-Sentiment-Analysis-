import pickle
import numpy as np
import tensorflow
import keras
from keras_bert import get_custom_objects
from keras import initializers


initializer = tensorflow.keras.initializers.GlorotNormal(seed=42)

features = pickle.load(open('tf_model.preproc','rb'))

new_model=keras.models.load_model('tf_model.h5', custom_objects=get_custom_objects())
data = 'the movie was trash'
pre_data = features.preprocess([data])
result = new_model.predict(pre_data)
print(result)
