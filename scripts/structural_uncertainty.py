# This script is used to calculate STRUCTURAL uncertainty for the smos-smap NN, and the corresponding mean ans std, as well as some other auxiliary data such as correlations, rmse, sliced and grouped data for plots

# input files: NNsmossmap_smos_output_ver1.pkl to ver7.pkl,
# NNsmossmap_overlap_output_ver1.pkl to ver7.pkl, sm_seas_cycle.pkl, overlap_full_with_NNoutput.pkl

# output files:
#  cor_between_SMSMAP_FINAL_2015_2020.pkl, cor_between_SMSMAP_RESID_FINAL_2015_2020.pkl
# VARS_2015-2020_FINAL_for_MAP.pkl
# rmse_resid_2015_2020_FINAL_for_MAP.pkl, rmse_full_2015_2020_FINAL_for_MAP.pkl, rmse_unbiased_2015_2020_FINAL_for_MAP.pkl
# stuct_unc_in_2010-2020.pkl
# VARS_2010-2020_May_for_MAP.pkl, VARS_2010-2020_May_for_ts.pkl
# smos1020_out_short_2002-2010_JULY_for_HOVEMOLLER.pkl

import sklearn as skl
from sklearn import preprocessing
from sklearn.metrics import  mean_squared_error
import pandas as pd
import numpy as np
import time
from datetime import timedelta



path_to_file = '~/'
path_to_save_output = '~/'

# read SM seasonal cycle
file_data = path_to_file + 'sm_seas_cycle.pkl'
sc = pd.read_pickle(file_data)

def combine_NN_outputs(file_name_, is_smap):

    # read all NN outputs and put in 1 dictionary

    num_of_outputs = 7 # how many NN runs
    dict_of_NNoutputs = {}
    
# here is defined by "file_name_" which group of files to read
    for i in range(1, num_of_outputs+1):
        filename = file_name_ + str(i) + '.pkl'
        initial = pd.read_pickle(path_to_file + filename)
        initial['date'] = pd.to_datetime(initial['date'])
        version = initial[['lat', 'lon', 'date', 'nn_out']].copy()
        version_name = 'ver' + str(i)
        dict_of_NNoutputs[version_name] = version

    #combine all runs
    ver1 = dict_of_NNoutputs['ver1']
    ver2 = dict_of_NNoutputs['ver2']

    combined_output = ver1.merge(ver2, on=['lat', 'lon', 'date'], suffixes=('_v1',  '_v2'))

    for i in range(3, num_of_outputs+1):
        version_name = 'ver' + str(i)
        ver_next =dict_of_NNoutputs[version_name]
        if (i % 2) == 0:
            suf1 = '_v' + str(i)
            suf2 = '_v' + str(i+2)
            combined_output = combined_output.merge(ver_next, on=['lat', 'lon', 'date'], suffixes=(suf1, suf2))
        else:
            combined_output = combined_output.merge(ver_next, on=['lat', 'lon', 'date'])


    combined_output['day'] = combined_output['date'].dt.dayofyear
    combined_output = pd.merge(combined_output, sc, how="left", on=['lat', 'lon', 'day'])

    cols = []
    for i in range(1, num_of_outputs+1):
        col_name ='nn_out_v' + str(i)
        cols.append(col_name)
    
# calculate mean and std over NN outputs

    combined_output['str_mean'] = combined_output[cols].mean(axis=1)

    combined_output['str_std'] = combined_output[cols].std(axis=1)

    combined_output['range_min'] = combined_output['str_mean'] - combined_output['str_std']
    combined_output['range_max'] = combined_output['str_mean'] + combined_output['str_std']

    #combined_output['range_2min'] = combined_output['str_mean'] - 2*combined_output['str_std']
    #combined_output['range_2max'] = combined_output['str_mean'] + 2*combined_output['str_std']

    combined_output = combined_output.drop(columns=cols)

    combined_output['covar'] = combined_output['str_std']/combined_output['str_mean']
    combined_output['resid_mean'] = combined_output['str_mean'] - combined_output['sm_am_seas_med']

# only calculate this if SMAP SM is available
    if is_smap:
        file_additioal = path_to_file+ 'overlap_full_with_NNoutput.pkl'
        additional = pd.read_pickle(file_additioal)

        combined_output = pd.merge(combined_output, file_additioal, on=['lat', 'lon', 'date'])
# for SMAP, more can be calculated
        combined_output['bias'] = combined_output['str_mean'] - combined_output['sm_am']
        cor = combined_output.groupby(['lat', 'lon'])[['sm_am','str_mean']].corr().iloc[0::2,-1]
        cor = cor.reset_index()
        cor.to_pickle(path_to_save_output + 'cor_between_SMSMAP_FINAL_2015_2020.pkl', protocol = 4)

        cor2 = combined_output.groupby(['lat', 'lon'])[['dev_sm','resid_mean']].corr().iloc[0::2,-1]
        cor2 = cor2.reset_index()
        cor2.to_pickle(path_to_save_output + 'cor_between_SMSMAP_RESID_FINAL_2015_2020.pkl', protocol = 4)
        
        
        combined_output_space = combined_output.groupby(['lat', 'lon'])['str_mean', 'str_std', 'covar', 'sm_am', 'bias'].mean()
        combined_output_space = combined_output_space.reset_index()
        combined_output_space.to_pickle(path_to_save_output + 'VARS_2015-2020_FINAL_for_MAP.pkl', protocol = 4)
        
        # defined as 3 functions since applied to group by and cannot take additinal arguments
        def rmse_res(g):
            rmse = np.sqrt( mean_squared_error( g['resid_mean'], g['dev_sm'] ) )
            return  rmse
        def rmse_ful(g):
            rmse = np.sqrt( mean_squared_error( g['str_mean'], g['sm_am'] ) )
            return  rmse
        def rmse_unbiased(g):
            rmse = np.sqrt( mean_squared_error( g['str_mean_ub'], g['sm_am_ub'] ) )
            return  rmse
            
        combined_output_short = combined_output[['lat', 'lon', 'date', 'str_mean', 'sm_am', 'resid_mean', 'dev_sm']]
        combined_output_short = combined_output_short.join(combined_output_short.groupby([ 'lat', 'lon'])['str_mean', 'sm_am', 'resid_mean', 'dev_sm'].mean(), on=[ 'lat', 'lon'], rsuffix='_aver')

        rmse_residual = combined_output.groupby(['lat', 'lon'] ).apply( rmse_res ).reset_index()
        rmse_full = combined_output.groupby(['lat', 'lon'] ).apply( rmse_ful ).reset_index()

        combined_output['str_mean_ub'] = combined_output_j['str_mean'] - combined_output['str_mean_aver']
        combined_output_j['sm_am_ub'] = combined_output_j['sm_am'] - combined_output_j['sm_am_aver']

        rmse_unbi = combined_output_j.groupby(['lat', 'lon'] ).apply( rmse_unbiased ).reset_index()

        rmse_residual.to_pickle(path_to_save_output + 'rmse_resid_2015_2020_FINAL_for_MAP.pkl', protocol = 4)
        rmse_full.to_pickle(path_to_save_output + 'rmse_full_2015_2020_FINAL_for_MAP.pkl', protocol = 4)
        rmse_unbi.to_pickle(path_to_save_output + 'rmse_unbiased_2015_2020_FINAL_for_MAP.pkl', protocol = 4)

    else:
        
        combined_output.to_pickle(path_to_save_output + 'stuct_unc_in_2010-2020.pkl', protocol = 4)

        combined_output_space = combined_output.groupby(['lat', 'lon'])['str_mean', 'str_std', 'covar'].mean()
        combined_output_space = combined_output_space.reset_index()
        combined_output_space.to_pickle( path_to_save_output + 'VARS_2010-2020_May_for_MAP.pkl', protocol = 4)

        combined_output_ts = combined_output.groupby(['date'])['str_mean', 'str_std', 'range_min', 'range_max'].mean()
        combined_output_ts = combined_output_ts.reset_index()
        combined_output_ts.to_pickle(path_to_save_output + 'VARS_2002-2010_May_for_ts.pkl', protocol = 4)

        combined_output_short = combined_output[['lat', 'lon', 'date', 'str_mean', 'resid_mean', 'sm_am_seas_med']].copy()
        combined_output_short.to_pickle(path_to_save_output + 'smos1020_out_short_2002-2010_JULY_for_HOVEMOLLER.pkl', protocol = 4)
        
    return

combine_NN_outputs(NNsmossmap_smos_output_ver, 0)

combine_NN_outputs(NNsmossmap_overlap_output_ver, 1)


quit()
