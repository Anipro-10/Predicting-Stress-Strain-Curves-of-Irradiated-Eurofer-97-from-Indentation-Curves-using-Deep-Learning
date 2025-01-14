import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing PH Datasets

PH250 = pd.read_csv('PH250.csv')
PH250 = PH250.iloc[:,0:2]
header_row = pd.DataFrame([PH250.columns.values], columns=PH250.columns)
PH250 = pd.concat([header_row, PH250]).reset_index(drop=True)
PH250.columns = ['P', 'H']
PH250 = np.array(PH250,dtype = 'float64')

PH300 = pd.read_csv('PH300.csv')
PH300 = PH300.iloc[:,0:2]
header_row = pd.DataFrame([PH300.columns.values], columns=PH300.columns)
PH300 = pd.concat([header_row, PH300]).reset_index(drop=True)
PH300.columns = ['P', 'H']
PH300 = np.array(PH300,dtype = 'float64')

PH350 = pd.read_csv('PH350.csv')
PH350 = PH350.iloc[:,0:2]
header_row = pd.DataFrame([PH350.columns.values], columns=PH350.columns)
PH350 = pd.concat([header_row, PH350]).reset_index(drop=True)
PH350.columns = ['P', 'H']
PH350 = np.array(PH350,dtype = 'float64')

PH400 = pd.read_csv('PH400.csv')
PH400 = PH400.iloc[:,0:2]
header_row = pd.DataFrame([PH400.columns.values], columns=PH400.columns)
PH400 = pd.concat([header_row, PH400]).reset_index(drop=True)
PH400.columns = ['P', 'H']
PH400 = np.array(PH400,dtype = 'float64')

PH450 = pd.read_csv('PH450.csv')
PH450 = PH450.iloc[:,0:2]
header_row = pd.DataFrame([PH450.columns.values], columns=PH450.columns)
PH450 = pd.concat([header_row, PH450]).reset_index(drop=True)
PH450.columns = ['P', 'H']
PH450 = np.array(PH450,dtype = 'float64')

PHRT250 = pd.read_csv('PHRT250.csv')
PHRT250 = PHRT250.iloc[:,0:2]
header_row = pd.DataFrame([PHRT250.columns.values], columns=PHRT250.columns)
PHRT250 = pd.concat([header_row, PHRT250]).reset_index(drop=True)
PHRT250.columns = ['P', 'H']
PHRT250 = np.array(PHRT250,dtype = 'float64')

PHRT300 = pd.read_csv('PHRT300.csv')
PHRT300 = PHRT300.iloc[:,0:2]
header_row = pd.DataFrame([PHRT300.columns.values], columns=PHRT300.columns)
PHRT300 = pd.concat([header_row, PHRT300]).reset_index(drop=True)
PHRT300.columns = ['P', 'H']
PHRT300 = np.array(PHRT300,dtype = 'float64')

#Importing SS datasets and removing timeframe column

SS250 = pd.read_csv('SS250.csv')
SS250 = SS250.iloc[:,1:3]
header_row = pd.DataFrame([SS250.columns.values], columns=SS250.columns)
SS250 = pd.concat([header_row, SS250]).reset_index(drop=True)
SS250.columns = ['P', 'H']
SS250 = np.array(SS250, dtype = 'float64')

SS300_1 = pd.read_csv('SS300_1.csv')
SS300_1 = SS300_1.iloc[:,1:3]
header_row = pd.DataFrame([SS300_1.columns.values], columns=SS300_1.columns)
SS300_1 = pd.concat([header_row, SS300_1]).reset_index(drop=True)
SS300_1.columns = ['Stress', 'Strain']
SS300_1 = np.array(SS300_1, dtype = 'float64')

SS300_2 = pd.read_csv('SS300_2.csv')
SS300_2 = SS300_2.iloc[:,1:3]
header_row = pd.DataFrame([SS300_2.columns.values], columns=SS300_2.columns)
SS300_2 = pd.concat([header_row, SS300_2]).reset_index(drop=True)
SS300_2.columns = ['Stress', 'Strain']
SS300_2 = np.array(SS300_2, dtype = 'float64')

SS350_1 = pd.read_csv('SS350_1.csv')
SS350_1 = SS350_1.iloc[:,1:3]
header_row = pd.DataFrame([SS350_1.columns.values], columns=SS350_1.columns)
SS350_1 = pd.concat([header_row, SS350_1]).reset_index(drop=True)
SS350_1.columns = ['Stress', 'Strain']
SS350_1 = np.array(SS350_1, dtype = 'float64')

SS350_2 = pd.read_csv('SS350_2.csv')
SS350_2 = SS350_2.iloc[:,1:3]
header_row = pd.DataFrame([SS350_2.columns.values], columns=SS350_2.columns)
SS350_2 = pd.concat([header_row, SS350_2]).reset_index(drop=True)
SS350_2.columns = ['Stress', 'Strain']
SS350_2 = np.array(SS350_2, dtype = 'float64')

SS400_1 = pd.read_csv('SS400_1.csv')
SS400_1 = SS400_1.iloc[:,1:3]
header_row = pd.DataFrame([SS400_1.columns.values], columns=SS400_1.columns)
SS400_1 = pd.concat([header_row, SS400_1]).reset_index(drop=True)
SS400_1.columns = ['Stress', 'Strain']
SS400_1 = np.array(SS400_1, dtype = 'float64')

SS400_2 = pd.read_csv('SS400_2.csv')
SS400_2 = SS400_2.iloc[:,1:3]
header_row = pd.DataFrame([SS400_2.columns.values], columns=SS400_2.columns)
SS400_2 = pd.concat([header_row, SS400_2]).reset_index(drop=True)
SS400_2.columns = ['Stress', 'Strain']
SS400_2 = np.array(SS400_2, dtype = 'float64')

SS450_1 = pd.read_csv('SS450_1.csv')
SS450_1 = SS450_1.iloc[:,1:3]
header_row = pd.DataFrame([SS450_1.columns.values], columns=SS450_1.columns)
SS450_1 = pd.concat([header_row, SS450_1]).reset_index(drop=True)
SS450_1.columns = ['Stress', 'Strain']
SS450_1 = np.array(SS450_1, dtype = 'float64')

SS450_2 = pd.read_csv('SS450_2.csv')
SS450_2 = SS450_2.iloc[:,1:3]
header_row = pd.DataFrame([SS450_2.columns.values], columns=SS450_2.columns)
SS450_2 = pd.concat([header_row, SS450_2]).reset_index(drop=True)
SS450_2.columns = ['Stress', 'Strain']
SS450_2 = np.array(SS450_2, dtype = 'float64')

SSRT250 = pd.read_csv('SSRT250.csv')
SSRT250 = SSRT250.iloc[:,1:3]
header_row = pd.DataFrame([SSRT250.columns.values], columns=SSRT250.columns)
SSRT250 = pd.concat([header_row, SSRT250]).reset_index(drop=True)
SSRT250.columns = ['Stress', 'Strain']
SSRT250 = np.array(SSRT250, dtype = 'float64')

SSRT300 = pd.read_csv('SSRT300.csv')
SSRT300 = SSRT300.iloc[:,1:3]
header_row = pd.DataFrame([SSRT300.columns.values], columns=SSRT300.columns)
SSRT300 = pd.concat([header_row, SSRT300]).reset_index(drop=True)
SSRT300.columns = ['Stress', 'Strain']
SSRT300 = np.array(SSRT300, dtype = 'float64')

#Plotting the PH data before pre-processing:

plt.plot(PH250[:,0], PH250[:, 1], color = 'red', label = 'PH250')
plt.plot(PH300[:,0], PH300[:, 1], color = 'green', label = 'PH300')
plt.plot(PH350[:,0], PH350[:, 1], color = 'blue', label = 'PH350')
plt.plot(PH400[:,0], PH400[:, 1], color = 'orange', label = 'PH400')
plt.plot(PH450[:,0], PH450[:, 1], color = 'purple', label = 'PH450')
plt.plot(PHRT250[:,0], PHRT250[:, 1], color = 'brown', label = 'PHRT250')
plt.plot(PHRT300[:,0], PHRT300[:, 1], color = 'black', label = 'PHRT300')

plt.ylabel('Force (N)')
plt.xlabel('Displacement (microns)')
plt.title('PH Curves')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(SS250[:,0], SS250[:, 1], color = 'red', label = 'SS250')
plt.plot(SS300_1[:,0], SS300_1[:, 1], color = 'green', label = 'SS300_1')
plt.plot(SS300_2[:,0], SS300_2[:, 1], color = 'blue', label = 'SS300_2')
plt.plot(SS350_1[:,0], SS350_1[:, 1], color = 'orange', label = 'SS350_1')
plt.plot(SS350_2[:,0], SS350_2[:, 1], color = 'purple', label = 'SS350_2')
plt.plot(SS400_1[:,0], SS400_1[:, 1], color = 'brown', label = 'SS400_1')
plt.plot(SS400_2[:,0], SS400_2[:, 1], color = 'black', label = 'SS400_2')
plt.plot(SS450_1[:,0], SS450_1[:, 1], color = 'cyan', label = 'SS450_1')
plt.plot(SS450_2[:,0], SS450_2[:, 1], color = 'magenta', label = 'SS450_2')
plt.plot(SSRT250[:,0], SSRT250[:, 1], color = 'pink', label = 'SSRT250')
plt.plot(SSRT300[:,0], SSRT300[:, 1], color = 'yellow', label = 'SSRT300')

plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Stress-Strain Curves')
plt.legend()
plt.grid(True)
plt.show()

def find_cutoff_index(data, cutoff):
    index = np.where(data[:,1] >= cutoff)[0]
    return index[0]

def shift_origin(data, index):
    data = data - data[index, :]
    return data

def remove_below_cutoff(data, index):
    data = data[index:, :]
    return data

def trim_at_max(data):
    index = np.argmax(data[:, 1])
    data_trimmed = data[:index+1, :]
    return data_trimmed

cutoff = 300

s250 = find_cutoff_index(SS250, cutoff) #find the first index of the all the datasets where the value stored is greater than cutoff
s300_1 = find_cutoff_index(SS300_1, cutoff)
s300_2 = find_cutoff_index(SS300_2, cutoff)
s350_1 = find_cutoff_index(SS350_1, cutoff)
s350_2 = find_cutoff_index(SS350_2, cutoff)
s400_1 = find_cutoff_index(SS400_1, cutoff)
s400_2 = find_cutoff_index(SS400_2, cutoff)
s450_1 = find_cutoff_index(SS450_1, cutoff)
s450_2 = find_cutoff_index(SS450_2, cutoff)
srt250= find_cutoff_index(SSRT250, cutoff)
srt300= find_cutoff_index(SSRT250, cutoff)

SS250 = shift_origin(SS250, s250) #shift the origin
SS300_1 = shift_origin(SS300_1, s300_1)
SS300_2 = shift_origin(SS300_2, s300_2)
SS350_1 = shift_origin(SS350_1, s350_1)
SS350_2 = shift_origin(SS350_2, s350_2)
SS400_1 = shift_origin(SS400_1, s400_1)
SS400_2 = shift_origin(SS400_2, s400_2)
SS450_1 = shift_origin(SS450_1, s450_1)
SS450_2 = shift_origin(SS450_2, s450_2)
SSRT250 = shift_origin(SSRT250, srt250)
SSRT300 = shift_origin(SSRT300, srt300)

SS250 = remove_below_cutoff(SS250, s250)
SS300_1 = remove_below_cutoff(SS300_1, s300_1)
SS300_2 = remove_below_cutoff(SS300_2, s300_2)
SS350_1 = remove_below_cutoff(SS350_1, s350_1)
SS350_2 = remove_below_cutoff(SS350_2, s350_2)
SS400_1 = remove_below_cutoff(SS400_1, s400_1)
SS400_2 = remove_below_cutoff(SS400_2, s400_2)
SS450_1 = remove_below_cutoff(SS450_2, s450_1)
SS450_2 = remove_below_cutoff(SS450_2, s450_2)
SSRT250 = remove_below_cutoff(SSRT250, srt250)
SSRT250 = remove_below_cutoff(SS250, srt300)

plt.plot(SS250[:,0], SS250[:, 1], color = 'red', label = 'SS250')
plt.plot(SS300_1[:,0], SS300_1[:, 1], color = 'green', label = 'SS300_1')
plt.plot(SS300_2[:,0], SS300_2[:, 1], color = 'blue', label = 'SS300_2')
plt.plot(SS350_1[:,0], SS350_1[:, 1], color = 'orange', label = 'SS350_1')
plt.plot(SS350_2[:,0], SS350_2[:, 1], color = 'purple', label = 'SS350_2')
plt.plot(SS400_1[:,0], SS400_1[:, 1], color = 'brown', label = 'SS400_1')
plt.plot(SS400_2[:,0], SS400_2[:, 1], color = 'black', label = 'SS400_2')
plt.plot(SS450_1[:,0], SS450_1[:, 1], color = 'cyan', label = 'SS450_1')
plt.plot(SS450_2[:,0], SS450_2[:, 1], color = 'magenta', label = 'SS450_2')
plt.plot(SSRT250[:,0], SSRT250[:, 1], color = 'pink', label = 'SSRT250')
plt.plot(SSRT300[:,0], SSRT300[:, 1], color = 'yellow', label = 'SSRT300')

plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Stress-Strain Curves')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(PH250[:,0], PH250[:, 1], color = 'red', label = 'PH250')
plt.plot(PH300[:,0], PH300[:, 1], color = 'green', label = 'PH300')
plt.plot(PH350[:,0], PH350[:, 1], color = 'blue', label = 'PH350')
plt.plot(PH400[:,0], PH400[:, 1], color = 'orange', label = 'PH400')
plt.plot(PH450[:,0], PH450[:, 1], color = 'purple', label = 'PH450')
plt.plot(PHRT250[:,0], PHRT250[:, 1], color = 'brown', label = 'PHRT250')
plt.plot(PHRT300[:,0], PHRT300[:, 1], color = 'black', label = 'PHRT300')

plt.ylabel('Force (N)')
plt.xlabel('Displacement (microns)')
plt.title('PH Curves')
plt.legend()
plt.grid(True)
plt.show()

import math

def smoothen(data):
    length = data.shape[0]
    cutsize = 99
    pieces = math.floor(length/cutsize)
    excess = length%cutsize
    smooth_data = np.zeros((100, data.shape[1]))
    cur = 0
    for i in range(99):
        smooth_data[i, :] = np.mean(data[cur:cur + pieces - 1, :], axis = 0)
        cur += pieces
    smooth_data[99, :] = np.mean(data[cur:, :], axis = 0)

    return smooth_data

#Smoothening PH Curves

smoothPH250 = smoothen(PH250)
smoothPH300 = smoothen(PH300)
smoothPH350 = smoothen(PH350)
smoothPH400 = smoothen(PH400)
smoothPH450 = smoothen(PH450)
smoothPHRT250 = smoothen(PHRT250)
smoothPHRT300 = smoothen(PHRT300)

smoothSS250 = smoothen(SS250)
smoothSS300_1 = smoothen(SS300_1)
smoothSS300_2 = smoothen(SS300_2)
smoothSS350_1 = smoothen(SS350_1)
smoothSS350_2 = smoothen(SS350_2)
smoothSS400_1 = smoothen(SS400_1)
smoothSS400_2 = smoothen(SS400_2)
smoothSS450_1 = smoothen(SS450_1)
smoothSS450_2 = smoothen(SS450_2)
smoothSSRT250 = smoothen(SSRT250)
smoothSSRT300 = smoothen(SSRT300)

#Plotting smoothened PH curves

plt.plot(smoothPH250[:,0], smoothPH250[:, 1], color = 'red', label = 'PH250')
plt.plot(smoothPH300[:,0], smoothPH300[:, 1], color = 'green', label = 'PH300')
plt.plot(smoothPH350[:,0], smoothPH350[:, 1], color = 'blue', label = 'PH350')
plt.plot(smoothPH400[:,0], smoothPH400[:, 1], color = 'orange', label = 'PH400')
plt.plot(smoothPH450[:,0], smoothPH450[:, 1], color = 'purple', label = 'PH450')
plt.plot(smoothPHRT250[:,0], smoothPHRT250[:, 1], color = 'brown', label = 'PHRT250')
plt.plot(smoothPHRT300[:,0], smoothPHRT300[:, 1], color = 'black', label = 'PHRT300')

plt.ylabel('Force (N)')
plt.xlabel('Displacement (microns)')
plt.title('Smoothened PH Curves')
plt.legend()
plt.grid(True)
plt.show()

#Plotting smoothened SS Curves

plt.plot(smoothSS250[:,0], smoothSS250[:, 1], color = 'red', label = 'SS250')
plt.plot(smoothSS300_1[:,0], smoothSS300_1[:, 1], color = 'green', label = 'SS300_1')
plt.plot(smoothSS300_2[:,0], smoothSS300_2[:, 1], color = 'blue', label = 'SS300_2')
plt.plot(smoothSS350_1[:,0], smoothSS350_1[:, 1], color = 'orange', label = 'SS350_1')
plt.plot(smoothSS350_2[:,0], smoothSS350_2[:, 1], color = 'purple', label = 'SS350_2')
plt.plot(smoothSS400_1[:,0], smoothSS400_1[:, 1], color = 'brown', label = 'SS400_1')
plt.plot(smoothSS400_2[:,0], smoothSS400_2[:, 1], color = 'black', label = 'SS400_2')
plt.plot(smoothSS450_1[:,0], smoothSS450_1[:, 1], color = 'cyan', label = 'SS450_1')
plt.plot(smoothSS450_2[:,0], smoothSS450_2[:, 1], color = 'magenta', label = 'SS450_2')
plt.plot(smoothSSRT250[:,0], smoothSSRT250[:, 1], color = 'pink', label = 'SSRT250')
plt.plot(smoothSSRT300[:,0], smoothSSRT300[:, 1], color = 'yellow', label = 'SSRT300')

plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Smoothened Stress-Strain Curves')
plt.legend()
plt.grid(True)
plt.show()

#Defining sequences to train LSTM network

seq = [
        smoothPH250.T,
        smoothPH300.T,
        smoothPH300.T,
        smoothPH350.T,
        smoothPH350.T,
        smoothPH400.T,
        smoothPH400.T,
        smoothPH450.T,
        smoothPH450.T,
        smoothPHRT250.T,
        smoothPHRT300.T
]

seqSS = [
    smoothSS250.T,
    smoothSS300_1.T,
    smoothSS300_2.T,
    smoothSS350_1.T,
    smoothSS350_2.T,
    smoothSS400_1.T,
    smoothSS400_2.T,
    smoothSS450_1.T,
    smoothSS450_2.T,
    smoothSSRT250.T,
    smoothSSRT300.T
]

features = pd.read_csv('Features.csv')
features = features.iloc[:,0:4]
features = np.array(features, dtype = 'float64')
features

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

train_index = [1,2,3,4,5,6,7,8,9,10]
test_index = [0]

features_train = features_scaled[train_index,:]
features_test = features_scaled[test_index, :]
if test_index[0] == 9:
    features_test = features_test*-1

Xtrain = []
Xtest = []
Ytrain = []
Ytest = []

k=0
features_train_rev = features_train
for i in train_index:
    Xtrain.append(seq[i])
    '''additional_features = np.tile(features_train[k, :], (seq[i].shape[0], 1))
    Xtrain.append(np.concatenate((seq[i], additional_features), axis = 1) )'''
    Ytrain.append(seqSS[i])
    k+=1


k=0
for i in test_index:
    Xtest.append(seq[i])
    '''additional_features = np.tile(features_test[k, :], (seq[i].shape[0], 1))
    Xtest.append(np.concatenate((seq[i], additional_features), axis = 1))'''
    Ytest.append(seqSS[i])
    k+=1

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)

seq = np.array(seq)
seqSS = np.array(seqSS)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain.reshape(Xtrain.shape[0], -1)).reshape(Xtrain.shape)
Xtest_scaled = scaler.transform(Xtest.reshape(Xtest.shape[0], -1)).reshape(Xtest.shape)

Xtrain_scaled = Xtrain_scaled.reshape(Xtrain.shape[0], Xtrain.shape[2], Xtrain.shape[1])
Xtest_scaled = Xtest_scaled.reshape(Xtest.shape[0], Xtest.shape[2], Xtest.shape[1])

Ytrain = Ytrain.reshape(Ytrain.shape[0], Ytrain.shape[2],Ytrain.shape[1])

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, LeakyReLU, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras import regularizers

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout, Dense, LeakyReLU, Input
from tensorflow.keras.models import Model

#Model

input1 = Input(shape = (Xtrain_scaled.shape[1], Xtrain_scaled.shape[2]))
input1 = BatchNormalization()(input1)
lstm = LSTM(units = 8)(input1)
l1 = Dense(units = 2, activation = 'tanh')(lstm) #make it two

input2 = Input(shape = (features_train.shape[1],))
l2 = Dense(units = 10, activation = 'tanh')(input2)
l3 = Dense(units = 10)(l2) #remove activation


merged = concatenate([l1, l3])
output = Dense(units = 200, activation = 'linear')(merged)

model4 = Model(inputs = [input1, input2], outputs = output)
model4.summary()

model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics = ['mae'])
model4.fit(x=[Xtrain_scaled, features_train], y=Ytrain.reshape(Ytrain.shape[0],-1), epochs=5000, batch_size=32) 

loss, mse = model4.evaluate(x=[Xtest_scaled, features_test], y=Ytest.reshape(Ytest.shape[0], -1))
print('Loss:', loss)
print('MSE:', mse)

# Predict
predictions = model4.predict(x=[Xtest_scaled, features_test])
predictions_reshaped = predictions.reshape(predictions.shape[0], 2, 100)

predictions_reshaped[0, 1, :] = np.abs(predictions_reshaped[0, 1, :])

plt.plot(predictions_reshaped[0, 0, :], predictions_reshaped[0, 1, :], color = 'red', label = 'Predicted')
plt.plot(Ytest[0, 0, :], Ytest[0, 1, :], color = 'blue', label = 'Actual')
plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Predicted vs Actual Stress-Strain Curve')
plt.legend()
plt.grid(True)
plt.show()

from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter
predictions_smoothed_x = savgol_filter(predictions_reshaped[0, 0, :], window_length=61, polyorder=2) #here
predictions_smoothed_y = savgol_filter(predictions_reshaped[0, 1, :], window_length=61, polyorder=2) #here

predictions_smoothed_x[0] = 0  
predictions_smoothed_y[0] = 0

Ytest_max_index = np.argmax(Ytest[0, 1, :])  # Index where Ytest[0, 1, :] is maximum
Ytest_max_x = Ytest[0, 0, Ytest_max_index]   # Corresponding x value
Ytest_max_y = Ytest[0, 1, Ytest_max_index]   # Maximum y value

# Find maximum values for smoothed predictions
predictions_max_index = np.argmax(predictions_smoothed_y)  # Index where predictions_smoothed_y is maximum    #here 
predictions_max_x = predictions_smoothed_x[predictions_max_index]  # Corresponding x value                    #here
predictions_max_y = predictions_smoothed_y[predictions_max_index]  # Maximum y value                          #here

# Print the coordinates of the maxima
print(f"x_observed = {Ytest_max_x}, Observed Maximm Stress = {Ytest_max_y}")
print(f"x_predicted = {predictions_max_x}, Predicted Maximum Stress= {predictions_max_y}")

# Plot the smoothed predictions
plt.plot(predictions_smoothed_x, predictions_smoothed_y, color='green', label='Predicted (Best Fit - Smoothed)') #here
plt.plot(Ytest[0, 0, :], Ytest[0, 1, :], color='blue', label='Actual')
plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Predicted vs Actual Stress-Strain Curve')
plt.legend()
plt.grid(True)
plt.show()






































































