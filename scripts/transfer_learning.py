# This script performs transfer learning

# input files: sm_seas_cycle.pkl, amsr_3dm.pkl, smos1020_out_short_2002-2010_JULY_for_HOVEMOLLER.pkl
# input preptained NN: amsr_smap_residual_v1.h5

# output files: new NN amsr_smap_transfer_1.h5, transfer_output_short_ver1.pkl

# auxiliary: amsre_3dm_seas_cycle.pkl


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
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import time
from datetime import timedelta


path_to_file = '~/'
path_to_save_nn = '~/'
path_to_save_output = '~/'

# read AMSRE data and add SM seasonal cycle
file_data = path_to_file + 'amsr_3dm.pkl'
fullamsre = pd.read_pickle(file_data)
fullamsre['date'] = pd.to_datetime(fullamsre['date'])
fullamsre = fullamsre[['date', 'lat', 'lon', 'day',
    'tb10_seas', 'tb18_seas', 'tb36_seas', 'tb89_seas',
    'tb10_resid','tb18_resid', 'tb36_resid', 'tb89_resid']]


file_data = path_to_file + 'sm_seas_cycle.pkl'
sc = pd.read_pickle(file_data)

fullamsre_sc = pd.merge(fullamsre, sc, how='left', on=['lat', 'lon', 'day'])
fullamsre_sc = fullamsre_sc.dropna()

fullamsre_sc.to_pickle(path_to_save_output + 'amsre_3dm_seas_cycle.pkl', protocol = 4)


# read NNsmossmap output (mean from structural uncertainty

file_data = path_to_file + 'smos1020_out_short_2002-2010_JULY_for_HOVEMOLLER.pkl'
target = pd.read_pickle(file_data)
target['date'] = pd.to_datetime(target['date'])


# overlap amsre with target SM data
overlap = pd.merge(fullamsre_sc, target, on=['date', 'lat', 'lon'], how = 'inner')
overlap = overlap.dropna()

# prepare data fror NN training
train = overlap.sample(frac=0.8)  #,random_state=200
test = overlap.drop(train.index)


train_dataset = train[[ 'lat', 'lon',  'tb10_resid', 'tb18_resid', 'tb36_resid', 'tb89_resid']]
test_dataset = test[[ 'lat', 'lon',  'tb10_resid', 'tb18_resid', 'tb36_resid', 'tb89_resid']]
full = overlap[[ 'lat', 'lon', 'tb10_resid', 'tb18_resid', 'tb36_resid', 'tb89_resid']].copy()
X_ofull = full.values.astype(float)
Y_ofull = overlap['str_mean'].values.astype(float)


X = train_dataset.values.astype(float) #scale.fit_transform(train_dataset)
Y = train['resid_mean'].values.astype(float) #scale.fit_transform(train_df['smap_sm'])

Xt = test_dataset.values.astype(float) #scale.fit_transform(train_dataset)
Yt = test['resid_mean'].values.astype(float) #scale.fit_transform(train_df['smap_sm'])

nn2 = fullamsre_sc[[ 'lat', 'lon', 'tb10_resid', 'tb18_resid', 'tb36_resid', 'tb89_resid']].copy()
scale =preprocessing.StandardScaler()

# scaling on the full possible data
scalerX = scale.fit(nn2)
X = scalerX.transform(X)
Xt = scalerX.transform(Xt)
X_ofull = scalerX.transform(X_ofull)
nn2 = scalerX.transform(nn2)

# NN parameters
bs = 1024
n_no_train = 3
adm = optimizers.Adam(lr=0.0008)
epoch=30

# read pretrained amsre-smap NN
model = keras.models.load_model(path_to_save_nn + 'amsr_smap_residual_v1.h5')

# set the first n_no_train layers non trainble
for layer in model.layers[:n_no_train]:
    layer.trainable = False


model.compile(loss='mean_squared_error', optimizer=adm, metrics=['mse'])
history = model.fit(X, Y, epochs=epoch, batch_size=bs, validation_split=0.2, verbose=2)

model.save(path_to_save_nn +  'amsr_smap_transfer_1.h5')

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



yt = model.predict(Xt, batch_size=bs, verbose=0)
yt = np.asarray(yt).reshape(-1)

Rt = np.corrcoef(Yt, yt)
R2 = r2_score(Yt, yt)
rmse = mean_squared_error(Yt, yt, squared=False)
print('Correlation R between the NN residuals and the target residuals for the Test part of the dataset')
print(Rt)
print('R^2 between the NN residuals and the target residuals for the Test part of the dataset')
print(R2)
print('RMSE between the NN residuals and the target residuals for the Test part of the dataset')
print(rmse)


# predict for full AMSR data starting 2002
nn2_out = model.predict(nn2, batch_size=bs, verbose=0)
fullamsre_sc['out_resid'] = nn2_out
fullamsre_sc['nn_out'] = fullamsre_sc['sm_am_seas_med'] + fullamsre_sc['out_resid']

short = fullamsre_sc[['date', 'lat', 'lon',  'nn_out']].copy()
short.to_pickle(path_to_save_output + 'transfer_output_short_ver1.pkl', protocol = 4)
