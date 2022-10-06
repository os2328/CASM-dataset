# This scripts gives an example of hyperparameters search with Sherpa

# input file: smap_smos_overlap_3dm_seas_c_no_std.pkl

# output file: trials' summary

import sklearn as skl
from sklearn import preprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import sherpa
import time
from datetime import timedelta



path_to_file = '~/'
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


features_smos = smos[[ 'lat', 'lon',  'dev_H', 'dev_V']]
X_smos = features_smos.values.astype(float)


scale =preprocessing.StandardScaler()
# fit scaler to all possible smos data
scalerX = scale.fit(X_smos)

X = scalerX.transform(X)
Xt = scalerX.transform(Xt)

# sherpa parameters to optimize
# learning rate lr - very important to optimize
# batch size
# number of neurons per layer num_units
# number of layers num_layers
parameters = [sherpa.Continuous(name='lr', range=[0.0001, 0.1], scale='log'),  sherpa.Ordinal(name='batch_size', range=[512, 1024, 2048, 4096]), sherpa.Discrete(name='num_units', range=[50, 1600]),  sherpa.Ordinal(name='num_layers', range=[5, 6, 7, 8])]

# some sherpa settings parameters
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=50)
study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=True)

# parameters search:
for trial in study:
    bs = trial.parameters['batch_size']
    model = Sequential()
    model.add(Dense(units=trial.parameters['num_units'], input_dim=X.shape[1], activation='relu'))
    model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    if trial.parameters['num_layers']==5:
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))        
    if trial.parameters['num_layers']==6:
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    if trial.parameters['num_layers']==7:
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    if rial.parameters['num_layers']==8:
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
        model.add(Dense(units=trial.parameters['num_units'], activation='relu'))
    model.add(Dense(1))
    adm = optimizers.Adam(lr=trial.parameters['lr'])
    model.compile(loss='mean_squared_error', optimizer=adm, metrics=['mse'])

# small number of epochs just to see the effect of the parameters - not getting to the full training
    epoch=7

    history = model.fit(X, Y, epochs=epoch, batch_size=bs, verbose=0, callbacks=[study.keras_callback(trial, objective_name='mean_squared_error')])

    study.finalize(trial)
    study.save(output_dir ='~/')

quit()

# analyze trial outputs to find the best parameters
