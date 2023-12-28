"""
Variables:

xtrain : To store features of training
xtest : To store features of testing
ytrain : To store target of training
ytest : To store target of testing
model : To store the ANN model object
early_stop : Object of early stopping
history : To store the metrics related to training of the model

Functions:

get_prediction: Predicts the sound from the filepath provided
    Arguments:
        filename: Path to the sound fileji
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from eda import lb as labelencoder
from eda import scaler
import librosa

# Reading relevant data
xtrain = pd.read_csv('xtrain.csv')
xtest = pd.read_csv('xtest.csv')
ytrain = pd.read_csv('ytrain.csv')
ytest = pd.read_csv('ytest.csv')

# Slicing the data for important features
xtrain = xtrain.iloc[:, 1:]
xtest = xtest.iloc[:, 1:]
ytrain = ytrain.iloc[:, 1:]
ytest = ytest.iloc[:, 1:]

# Creating model architecture
model = Sequential()
model.add(Dense(units = 128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Setting up a callback to avoid overfitting
early_stop = EarlyStopping(monitor='accuracy', patience=300, min_delta=0, mode='auto', verbose=0, restore_best_weights=True)

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(xtrain, ytrain, epochs=100, batch_size=32, callbacks=[early_stop])

# Function to get prediction on the new file
def get_prediction(filename):
    audio, sample_rate = librosa.load(filename) 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=129)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    predicted_label = [list(i).index(max(i)) for i in predicted_label]
    prediction_class = lb.inverse_transform(predicted_label) 
    return prediction_class