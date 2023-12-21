'''
Variables:

dataset : To create an object of audio dataset
filename : To store the path to audio file
data : To store the audio data
sample_rate : To store the rate at which audio data is sampled
audio_dataset_path : To store the path to directory where audio data is stored
metadata : To store the metadata of audio data as a DataFrame
data : To store the audio data
mfccs_features : To calculate MFCCS features of the data
mfccs_scaled_features : To scale the MFCCS features
extracted_features : To store the extracted features of the audio file
X : To store the features 
y : To store the target 
df_x : To store the features as a DataFrame
df_y : To store the target as a DataFrame

Functions:

features_extractor : Accepts file path of audio data as an argument and returns the MFCCS features of the same.
    Arguments: 
    file : Path to the audio file

'''

# Importing relevant libraries
import soundata # To download sound data
import matplotlib.pyplot as plt # To visualize data
%matplotlib inline
import IPython.display as ipd # To display audio to be played
import librosa # To process the audio data
import librosa.display # To visualize audio's waveplot
import pandas as pd # To preprocess the data
import os # To generate the path for data
import numpy as np # To perform numerical operations
from tqdm import tqdm # To display progress of the task

# Initialize dataset and set the location for audio data on system
dataset = soundata.initialize('urbansound8k', data_home='D:\Jingle Bytes')
# Download the dataset to the specified location
dataset.download()

# Select one file for understanding the data
filename = 'D:/Jingle Bytes/audio/fold1/101415-3-0-2.wav'

'''# Setting the figure size to display audio data
plt.figure(figsize=(14,5))
# Reading the data in sample audio file
data, sample_rate = librosa.load(filename)
# Displaing the waveplot of audio 
librosa.display.waveshow(data, sr=sample_rate)
# Displaying the audio so that it can be played.
ipd.Audio(filename)
'''

# Set the directory path for audio data
audio_dataset_path = 'D:/Jingle Bytes/audio'
# Read the metadata file as a DataFrame
metadata = pd.read_csv('D:/Jingle Bytes/metadata/UrbanSound8K.csv')

# Extract MFCCS features from a given audio file
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr = sample_rate, n_mfcc=400)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    return mfccs_scaled_features

# Initialize an empty list to extract features
extracted_features = []

# Iterate over the rows in metadata of audio files
for index_num, row in tqdm(metadata.iterrows()):
    # Generate the file name for each
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    # Generate the value of target variable for each
    final_class_labels=row["class"]
    # Create MFCCS features of the audio file
    data=features_extractor(file_name)
    # Add these features and target to the list
    extracted_features.append([data,final_class_labels])

# Convert the list of extracted features of all audio files to DataFrame
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])

# Split the resultant DataFrame into features (X) and target (y)
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

# Convert the features into a separate DataFrame
df_x = pd.DataFrame(X)
# Save the features as a csv file
df_x.to_csv('features.csv')
# Convert the target into a separate DataFrame
df_y = pd.DataFrame(y)
# Save the target as a csv file
df_y.to_csv('target.csv')
