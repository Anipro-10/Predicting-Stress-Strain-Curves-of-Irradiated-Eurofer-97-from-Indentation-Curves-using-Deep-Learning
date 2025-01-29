import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

UPH250 = pd.read_csv('unirrph250.csv')
UPH250 = UPH250.iloc[:,0:2]
header_row = pd.DataFrame([UPH250.columns.values], columns=UPH250.columns)
UPH250 = pd.concat([header_row, UPH250]).reset_index(drop=True)
UPH250.columns = ['P', 'H']
UPH250 = np.array(UPH250,dtype = 'float64')

UPH300 = pd.read_csv('unirrph300.csv')
UPH300 = UPH300.iloc[:,0:2]
header_row = pd.DataFrame([UPH300.columns.values], columns=UPH300.columns)
UPH300 = pd.concat([header_row, UPH300]).reset_index(drop=True)
UPH300.columns = ['P', 'H']
UPH300 = np.array(UPH300,dtype = 'float64')

UPH350 = pd.read_csv('unirrph350.csv')
UPH350 = UPH350.iloc[:,0:2]
header_row = pd.DataFrame([UPH350.columns.values], columns=UPH350.columns)
UPH350 = pd.concat([header_row, UPH350]).reset_index(drop=True)
UPH350.columns = ['P', 'H']
UPH350 = np.array(UPH350,dtype = 'float64')

UPHRT = pd.read_csv('unirrphrt.csv')
UPHRT = UPHRT.iloc[:,0:2]
header_row = pd.DataFrame([UPHRT.columns.values], columns=UPHRT.columns)
UPHRT = pd.concat([header_row, UPHRT]).reset_index(drop=True)
UPHRT.columns = ['P', 'H']
UPHRT = np.array(UPHRT,dtype = 'float64')

#Importing SS datasets

USS250 = pd.read_csv('unirr250.csv')
USS250 = USS250.iloc[:,0:2]
header_row = pd.DataFrame([USS250.columns.values], columns=USS250.columns)
USS250 = pd.concat([header_row, USS250]).reset_index(drop=True)
USS250.columns = ['P', 'H']
USS250 = np.array(USS250, dtype = 'float64')

USS300 = pd.read_csv('unirr300.csv')
USS300 = USS300.iloc[:,0:2]
header_row = pd.DataFrame([USS300.columns.values], columns=USS300.columns)
USS300 = pd.concat([header_row, USS300]).reset_index(drop=True)
USS300.columns = ['P', 'H']
USS300 = np.array(USS300, dtype = 'float64')

USS350 = pd.read_csv('unirr350.csv')
USS350 = USS350.iloc[:,0:2]
header_row = pd.DataFrame([USS350.columns.values], columns=USS350.columns)
USS350 = pd.concat([header_row, USS350]).reset_index(drop=True)
USS350.columns = ['P', 'H']
USS350 = np.array(USS350, dtype = 'float64')

USSRT = pd.read_csv('unirr-rt2.csv')
USSRT = USSRT.iloc[:,0:2]
header_row = pd.DataFrame([USSRT.columns.values], columns=USSRT.columns)
USSRT = pd.concat([header_row, USSRT]).reset_index(drop=True)
USSRT.columns = ['P', 'H']
USSRT = np.array(USSRT, dtype = 'float64')

#Plotting the PH data before pre-processing:

plt.plot(UPH250[:,0], UPH250[:, 1], color = 'red', label = 'UPH250')
plt.plot(UPH300[:,0], UPH300[:, 1], color = 'green', label = 'UPH300')
plt.plot(UPH350[:,0], UPH350[:, 1], color = 'blue', label = 'UPH350')
plt.plot(UPHRT[:,0], UPHRT[:, 1], color = 'orange', label = 'UPHRT')
plt.ylabel('Force (N)')
plt.xlabel('Displacement (microns)')
plt.title('PH Curves')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(USS250[:,0], USS250[:, 1], color = 'red', label = 'USS250')
plt.plot(USS300[:,0], USS300[:, 1], color = 'green', label = 'USS300')
plt.plot(USS350[:,0], USS350[:, 1], color = 'blue', label = 'USS350')
plt.plot(USSRT[:,0], USSRT[:, 1], color = 'orange', label = 'USSRT')

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

#find the first index of the all the datasets where the value stored is greater than cutoff
us250 = find_cutoff_index(USS250, cutoff)
us300 = find_cutoff_index(USS300, cutoff)
us350 = find_cutoff_index(USS350, cutoff)
usrt = find_cutoff_index(USSRT, cutoff)

#Shift the origin

USS250 = shift_origin(USS250, us250)
USS300 = shift_origin(USS300, us300)
USS350 = shift_origin(USS350, us350)
USSRT = shift_origin(USSRT, usrt)

USS250 = remove_below_cutoff(USS250, us250)
USS300 = remove_below_cutoff(USS300, us300)
USS350 = remove_below_cutoff(USS350, us350)
USSRT = remove_below_cutoff(USSRT, usrt)

plt.plot(USS250[:,0], USS250[:, 1], color = 'red', label = 'USS250')
plt.plot(USS300[:,0], USS300[:, 1], color = 'green', label = 'USS300')
plt.plot(USS350[:,0], USS350[:, 1], color = 'blue', label = 'USS350')
plt.plot(USSRT[:,0], USSRT[:, 1], color = 'orange', label = 'USSRT')

plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Stress-Strain Curves')
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

#Smoothening PH and SS Curves
smoothUPH250 = smoothen(UPH250)
smoothUPH300 = smoothen(UPH300)
smoothUPH350 = smoothen(UPH350)
smoothUPHRT = smoothen(UPHRT)

smoothUSS250 = smoothen(USS250)
smoothUSS300 = smoothen(USS300)
smoothUSS350 = smoothen(USS350)
smoothUSSRT = smoothen(USSRT)

plt.plot(smoothUPH250[:,0], smoothUPH250[:, 1], color = 'red', label = 'UPH250')
plt.plot(smoothUPH300[:,0], smoothUPH300[:, 1], color = 'green', label = 'UPH300')
plt.plot(smoothUPH350[:,0], smoothUPH350[:, 1], color = 'blue', label = 'UPH350')
plt.plot(smoothUPHRT[:,0], smoothUPHRT[:, 1], color = 'orange', label = 'UPHRT')

plt.ylabel('Force (N)')
plt.xlabel('Displacement (microns)')
plt.title('Smoothened PH Curves')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(smoothUSS250[:,0], smoothUSS250[:, 1], color = 'red', label = 'USS250')
plt.plot(smoothUSS300[:,0], smoothUSS300[:, 1], color = 'green', label = 'USS300')
plt.plot(smoothUSS350[:,0], smoothUSS350[:, 1], color = 'blue', label = 'USS350')
plt.plot(smoothUSSRT[:,0], smoothUSSRT[:, 1], color = 'orange', label = 'USSRT')

plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Smoothened Stress-Strain Curves')
plt.legend()
plt.grid(True)
plt.show()

#Defining sequences to train LSTM network
useq = [
        smoothUPH250.T,
        smoothUPH250.T,
        smoothUPH300.T,
        smoothUPH300.T,
        smoothUPH350.T,
        smoothUPH350.T,
        smoothUPHRT.T,
        smoothUPHRT.T
]

useqSS = [
    smoothUSS250.T,
    smoothUSS250.T,
    smoothUSS300.T,
    smoothUSS300.T,
    smoothUSS350.T,
    smoothUSS350.T,
    smoothUSSRT.T,
    smoothUSSRT.T
]

features = pd.read_csv('features2.csv')
features = features.iloc[:,0:4]
features = np.array(features, dtype = 'float64')
features

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

train_index = [2,3,4,5,6,7]
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
    Xtrain.append(useq[i])
    '''additional_features = np.tile(features_train[k, :], (seq[i].shape[0], 1))
    Xtrain.append(np.concatenate((seq[i], additional_features), axis = 1) )'''
    Ytrain.append(useqSS[i])
    k+=1


k=0
for i in test_index:
    Xtest.append(useq[i])
    '''additional_features = np.tile(features_test[k, :], (seq[i].shape[0], 1))
    Xtest.append(np.concatenate((seq[i], additional_features), axis = 1))'''
    Ytest.append(useqSS[i])
    k+=1

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)

useq = np.array(useq)
useqSS = np.array(useqSS)

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

input1 = Input(shape = (Xtrain_scaled.shape[1], Xtrain_scaled.shape[2]))
input1 = BatchNormalization()(input1)
lstm = LSTM(units = 8)(input1)
l1 = Dense(units = 2, activation = 'tanh')(lstm) #make it two

input2 = Input(shape = (features_train.shape[1],))
l2 = Dense(units = 10, activation = 'tanh')(input2)
l3 = Dense(units = 10)(l2) #remove activation


merged = concatenate([l1, l3])
l4 = Dense(100, 'tanh')(merged)
output = Dense(units = 200, activation = 'linear')(l4)

model4 = Model(inputs = [input1, input2], outputs = output)
model4.summary()

model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics = ['mae'])
model4.fit(x=[Xtrain_scaled, features_train], y=Ytrain.reshape(Ytrain.shape[0],-1), epochs=480, batch_size=32) 

loss, mse = model4.evaluate(x=[Xtest_scaled, features_test], y=Ytest.reshape(Ytest.shape[0], -1))
print('Loss:', loss)
print('MSE:', mse)

predictions = model4.predict(x=[Xtest_scaled, features_test])
predictions = np.abs(predictions)

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

predictions_smoothed_x_250_2 = savgol_filter(predictions_reshaped[0, 0, :], window_length=51, polyorder=4) #here
predictions_smoothed_y_250_2 = savgol_filter(predictions_reshaped[0, 1, :], window_length=51, polyorder=4) #here

#predictions_smoothed_x_250[0] = 0  
#predictions_smoothed_y_250[0] = 0

Ytest_max_index = np.argmax(Ytest[0, 1, :])  
Ytest_max_x = Ytest[0, 0, Ytest_max_index]  
Ytest_max_y = Ytest[0, 1, Ytest_max_index]   

# Find maximum values for smoothed predictions
predictions_max_index = np.argmax(predictions_smoothed_y_250_2)  
predictions_max_x = predictions_smoothed_x_250_2[predictions_max_index]  
predictions_max_y = predictions_smoothed_y_250_2[predictions_max_index]

# Print the coordinates of the maxima
print(f"x_observed = {Ytest_max_x}, Observed Maximm Stress = {Ytest_max_y}")
print(f"x_predicted = {predictions_max_x}, Predicted Maximum Stress= {predictions_max_y}")

plt.plot(predictions_smoothed_x_250_2, predictions_smoothed_y_250_2, color='green', label='Predicted (Best Fit - Smoothed)') #here
plt.plot(Ytest[0, 0, :], Ytest[0, 1, :], color='blue', label='Actual')
plt.ylabel('Force (N)')
plt.xlabel('Channel Length (mm)')
plt.title('Predicted vs Actual Stress-Strain Curve')
plt.legend()
plt.grid(True)
plt.show()









