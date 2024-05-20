import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import customtkinter

data_train = pd.read_excel('train2.xlsx', dtype = str)
data_test = pd.read_excel('test2.xlsx', dtype = str)

(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train, text_column = 'Reviews',label_columns = 'Sentiment', val_df = data_test,maxlen = 500, preprocess_mode = 'bert')

model = text.text_classifier(name = 'bert', train_data = (X_train, y_train), preproc = preproc)

learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),val_data = (X_test, y_test), batch_size = 6)


learner.fit_onecycle(lr = 2e-5, epochs = 1)

predictor = ktrain.get_predictor(learner.model, preproc)

data = ['this movie was horrible, the plot was really boring. acting was okay',
        'the fild is really sucked. there is not plot and acting was bad',
        'what a beautiful movie. great plot. acting was good. will see it again']

predictor.predict(data)
