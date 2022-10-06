# This script trains SMAP-SMOS NN

# input files: smap_smos_overlap_3dm_seas_c_no_std.pkl, smos_3dm_seas_c_no_std.pkl
# output files: saved NN NNsmapsmos_ver1.h5, NNsmossmap_overlap_output_ver1.pkl, NNsmossmap_smos_output_ver1.pkl, overlap_full_with_NNoutput.pkl

import sklearn as skl
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import time
from datetime import timedelta


path_to_file = '~/'
path_to_save_nn = '~/'
path_to_save_output = '~/'
file_data = path_to_file + 'smos_3dm_seas_c_no_std.pkl'
smos = pd.read_pickle(file_data)

file_data = path_to_file + 'smap_smos_overlap_3dm_seas_c_no_std.pkl'
overlap = pd.read_pickle(file_data)
overlap['date'] = pd.to_datetime(overlap['date'])
overlap = overlap.dropna()

# divide data into train and test
train = overlap.sample(frac=0.8)
test = overlap.drop(train.index)

# features are coordinates and TB residuals
train_dataset = train[[ 'lat', 'lon',  'dev_H', 'dev_V']]
test_dataset = test[[ 'lat', 'lon',  'dev_H', 'dev_V']]

X = train_dataset.values.astype(float)
# target is SM residuals
Y = train['dev_sm'].values.astype(float)

Xt = test_dataset.values.astype(float)
Yt = test['dev_sm'].values.astype(float)

X_ovelap_full = overlap[[ 'lat', 'lon',  'dev_H', 'dev_V']]
X_ovelap_full = X_ovelap_full.values.astype(float)
Y_ovelap_full = overlap['dev_sm'].values.astype(float)

features_smos = smos[[ 'lat', 'lon',  'dev_H', 'dev_V']]
features_smos = features_smos.dropna() # don't do drop na before subsetting only 1 angle, otherwise will wrongly drop data due to many na in other angles.

X_smos = features_smos.values.astype(float)

scale =preprocessing.StandardScaler()
# fit scaler to all possible smos data
scalerX = scale.fit(X_smos)

X = scalerX.transform(X)
Xt = scalerX.transform(Xt)
X_ovelap_full = scalerX.transform(X_ovelap_full)
X_smos = scalerX.transform(X_smos)

# NN parameters
bs = 1024
num_of_units = 1050
adm = optimizers.Adam(lr=0.0002)
epoch=30

inputs = tf.keras.layers.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(units=num_of_units,  activation='relu')(inputs)
x = tf.keras.layers.Dense(units=num_of_units,  activation='relu')(x)
x = tf.keras.layers.Dense(units=num_of_units,  activation='relu')(x)
x = tf.keras.layers.Dense(units=num_of_units, activation='relu')(x)
x = tf.keras.layers.Dense(units=num_of_units,  activation='relu')(x)
x = tf.keras.layers.Dense(units=num_of_units,  activation='relu')(x)
x = tf.keras.layers.Dense(units=num_of_units,  activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(loss='mean_squared_error', optimizer=adm, metrics=['mse'])
history = model.fit(X, Y, epochs=epoch, batch_size=bs, validation_split=0.2, verbose=2)
model.save(path_to_save_nn +'NNsmapsmos_ver1.h5')

#print(model.summary())

print('NN performance metrics')

y = model.predict(X, batch_size=bs, verbose=0)
y = np.asarray(y).reshape(-1)
R = np.corrcoef(Y, y)
R2 = r2_score(Y, y)
rmse = mean_squared_error(Y, y, squared=False)
print('Correlation R between the NN residuals and the target residuals for the Training part of the dataset')
print(R)
print('R^2 between the NN residuals and the target residuals for the Training part of the dataset')
print(R2)
print('RMSE between the NN residuals and the target residuals for the Training part of the dataset')
print(rmse)

y_t = model.predict(Xt, batch_size=bs, verbose=0)
y_t = np.asarray(y_t).reshape(-1)
R = np.corrcoef(Yt, y_t)
R2 = r2_score(Yt, y_t)
rmse = mean_squared_error(Yt, y_t, squared=False)
print('Correlation R between the NN residuals and the target residuals for the Test part of the dataset')
print(R)
print('R^2 between the NN residuals and the target residuals for the Test part of the dataset')
print(R2)
print('RMSE between the NN residuals and the target residuals for the Test part of the dataset')
print(rmse)

y_f = model.predict(X_ovelap_full, batch_size=bs, verbose=0)
y_f = np.asarray(y_f).reshape(-1)
overlap['out_resid'] = y_f
overlap['nn_out'] = overlap['sm_am_seas_med'] + overlap['out_resid']
y_ovelap_full = overlap['nn_out'].values.astype(float)
R = np.corrcoef(Y_ovelap_full, y_ovelap_full)
R2 = r2_score(Y_ovelap_full, y_ovelap_full)
rmse = mean_squared_error(Y_ovelap_full, y_ovelap_full, squared=False)
print('Correlation R between the NN full SM signal and the target full SM signal for the whole period 2015-2020')
print(R)
print('R^2 between the NN full SM signal and the target full SM signal for the whole period 2015-2020')
print(R2)
print('RMSE between the NN full SM signal and the target full SM signal for the whole period 2015-2020')
print(rmse)


overlap_ver = overlap[['lat', 'lon', 'date', 'nn_out']].copy()

overlap_ver.to_pickle(path_to_save_output + 'NNsmossmap_overlap_output_ver1.pkl', protocol = 4)
overlap.to_pickle(path_to_save_output + 'overlap_full_with_NNoutput.pkl', protocol = 4)

y_smos = model.predict(X_smos, batch_size=bs, verbose=0)
y_smos = np.asarray(y_smos).reshape(-1)
smos['output_resid'] = y_smos
smos['nn_out'] = smos['sm_am_seas_med']+smos1015['output_resid']

smos.to_pickle(path_to_save_output + 'NNsmossmap_smos_output_ver1.pkl', protocol = 4)


