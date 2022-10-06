# This script performs forward runs using saved NN and noisy input data to calculate data-related uncertainty for 1 (one) pretrained NN

# Input files: 'amsre_3dm_seas_cycle.pkl'
# pretrained NN: amsr_smap_transfer_1.h5

# Output files: data_uncer_amsre_ver1.pkl


import sklearn as skl
from sklearn import preprocessing
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
from tensorflow.keras.models import load_model

#from matplotlib import pyplot
#import sherpa
import time
from datetime import timedelta
#import tensorflow.keras.backend as K



path_to_file = '~/'
path_to_save_nn = '~/'
path_to_save_output = '~/'

# read AMSR file
file_data = path_to_file + 'amsre_3dm_seas_cycle.pkl'
amsre = pd.read_pickle(file_data)
amsre['date'] = pd.to_datetime(amsre['date'])
amsre = amsre.dropna()
amsre = amsre.sort_values(by = ['lat', 'lon', 'date'])

# calculate residuals standard deviation per location

amsre_std= amsre.join(amsre.groupby(['lat', 'lon'])[ 'tb10_resid', 'tb18_resid', 'tb36_resid', 'tb89_resid'].std(), on=['lat', 'lon'], rsuffix='_std_per_loc')


# define noise functions (separate functions per column since the function is applied to grouped data and doesn't take arguments

noise_level = 0.1
def noise_fun_10(row):
    std = row['tb10_resid_std_per_loc']
    return np.random.normal(0, noise_level*std, 1)
def noise_fun_18(row):
    std = row['tb18_resid_std_per_loc']
    return np.random.normal(0, noise_level*std, 1)
def noise_fun_36(row):
    std = row['tb36_resid_std_per_loc']
    return np.random.normal(0, noise_level*std, 1)
def noise_fun_89(row):
    std = row['tb89_resid_std_per_loc']
    return np.random.normal(0, noise_level*std, 1)

# load pretrained NN
model = keras.models.load_model(path_to_save_nn + 'amsr_smap_transfer_1.h5')

# NN parameters
bs = 1024

flag = 1

number_of_forvard_runs = 10

for i in range(1, number_of_forvard_runs):

    # calculate noise
    amsre_std['10_noise'] = amsre_std.apply(lambda row: noise_fun_10(row), axis=1)
    amsre_std['18_noise'] = amsre_std.apply(lambda row: noise_fun_18(row), axis=1)
    amsre_std['36_noise'] = amsre_std.apply(lambda row: noise_fun_36(row), axis=1)
    amsre_std['89_noise'] = amsre_std.apply(lambda row: noise_fun_89(row), axis=1)

# add noise to the input
    amsre_std['noisy_tb10'] = amsre_std['10_noise'] + amsre_std['tb10_resid']
    amsre_std['noisy_tb18'] = amsre_std['18_noise'] + amsre_std['tb18_resid']
    amsre_std['noisy_tb36'] = amsre_std['36_noise'] + amsre_std['tb36_resid']
    amsre_std['noisy_tb89'] = amsre_std['89_noise'] + amsre_std['tb89_resid']

# short for NN
    noisy_input = amsre_std[[ 'lat', 'lon', 'noisy_tb10', 'noisy_tb18', 'noisy_tb36', 'noisy_tb89']]

    X_f = noisy_input.values.astype(float)
    scale =preprocessing.StandardScaler()
    scalerX = scale.fit(X_f)
    X_f = scalerX.transform(X_f)


    nn_output_from_noisy = model.predict(X_f, batch_size=bs, verbose=0)
    nn_output_from_noisy = np.asarray(nn_output_from_noisy).reshape(-1)

    amsre_std['output_resid'] = nn_output_from_noisy

    amsre_std['nn_out'] = amsre_std['sm_am_seas_med']+amsre_std['output_resid']

    amsre_ver = amsre_std[['lat', 'lon', 'date', 'nn_out']].copy()

# Combine all outputs in 1 dataframe with version name
    if i ==1:
        amsre_out = amsre_ver.copy()
        continue

    if flag:
        namel = 'd' + str(i-1)
        namer = 'd' + str(i)

        amsre_out = amsre_out.merge(amsre_ver, on=['lat', 'lon', 'date'], suffixes=(namel,namer) )
        flag =0
    else:
        amsre_out = amsre_out.merge(amsre_ver, on=['lat', 'lon', 'date'])
        flag=1

amsre_out.to_pickle(path_to_save_output + 'data_uncer_amsre_ver1.pkl', protocol = 4)

