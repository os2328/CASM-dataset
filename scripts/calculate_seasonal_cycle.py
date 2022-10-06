# This script is used to calculate seasonal cycle for SMAP SM and SMOS TB data

# input files: smap_smos_overlap_3dm.pkl, smos_3dm.pkl

# output files: smap_smos_overlap_3dm_seas_c.pkl, smap_smos_overlap_3dm_seas_c_no_std.pkl,  smos_3dm_seas_c.pkl, smos_3dm_seas_c_no_std.pkl, sm_seas_cycle.pkl


import pandas as pd
import numpy as np
import time
from datetime import timedelta
from scipy.optimize import curve_fit
import math

# Read files
data_path = '~/'
file_data = data_path +'smap_smos_overlap_3dm.pkl'

overlap = pd.read_pickle(file_data)
overlap['date'] = pd.to_datetime(overlap['date'])
print(overlap.columns)
# only take incidence angle 42.5
overlap = overlap[['date', 'lat', 'lon', 'sm_am', 'bth10', 'tbv10']]
overlap['doy'] =  overlap['date'].dt.dayofyear

print('shape before seas cycle')
print(overlap.shape)

file_data_smos = data_path + 'smos_3dm.pkl'
smos_20102020 = pd.read_pickle(file_data_smos)
smos_20102020['date'] = pd.to_datetime(smos_20102020['date'])
# only take incidence angle 42.5
smos = smos_20102020[['date', 'lat', 'lon', 'bth10', 'tbv10']]
smos['doy'] =  smos['date'].dt.dayofyear


# Define functions that will calculate seasonal cycle. The functions are applied with group_by, for that reason they cannot take any other arguments, hence there are separate functions per each column for which it should be calculated.

# minimum number of datapoints to calculate seasonal cycle
n_min =40
def seas_cycle_H(df):
    global count_c # counts the number of points, at which seasonal cycle is calculated
    global count_m # counts the number of points, at which curve_fit could not be fit
    global count_l # counts the number of points, at which time series are too short (< n_min)
    t = df['t']
    y = df['tbh10']
    # simple sine wave
    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    # only apply if there are at least n_min values
    if len(y)<n_min:
    # if less, median of the values is used
        y_s =np.median(y)*y/y
        count_l = count_l+1

    else:
        try:
        # bounds for TB in K, period ~ 1 year - between 365 and 366 days
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 150. ], [40., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s =np.median(y)*y/y
            count_m = count_m+1
    return y_s


# same for V-polarization TB
def seas_cycle_V(df):
    global count_c
    global count_m
    global count_l
    t = df['t']
    y = df['tbv10']

    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    if len(y)<n_min:
        y_s =np.median(y)*y/y
        count_l = count_l+1
    else:

        try:
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 150. ], [40., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s = np.median(y)*y/y
            count_m = count_m+1
    return y_s

# save for SM, but the BOUNDS are different!
def seas_cycle(df):
    global count_c
    global count_m
    global count_l
    global seas_df
    t = df['t']
    y = df['sm_am']
    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    if len(y)<n_min:
        y_s =np.median(y)*y/y
        count_l = count_l+1
    else:
        try:
            popt, pcov = curve_fit(func, t, y, bounds=([0., 365., 0., 0. ], [0.4, 366., 2.0*np.pi, 0.6]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s =np.median(y)*y/y
            count_m = count_m+1
    return y_s



overlap = overlap.sort_values(by = ['date', 'lat', 'lon'])

# create time variable for curve_fit
d0 = overlap['date'].iloc[0]
overlap['t']  = (overlap['date']-d0)
overlap['t']  = overlap['t'].dt.days.astype('int16')

count_m = 0
count_l = 0
count_c = 0

# group by location and apply seasonal cycle function to each
grouped_overlap = overlap.groupby(['lat', 'lon'])
tbh_output = grouped_overlap.apply(seas_cycle_H).reset_index()
tbv_output= grouped_overlap.apply(seas_cycle_V).reset_index()
sm_output = grouped_overlap.apply(seas_cycle).reset_index()

tbh_output= tbh_output.set_index('level_2')
tbv_output= tbv_output.set_index('level_2')
sm_output= sm_output.set_index('level_2')

tbh_output = tbh_output.drop(columns=['lat', 'lon'])
tbv_output = tbv_output.drop(columns=['lat', 'lon'])
sm_output = sm_output.drop(columns=['lat', 'lon'])


tbh_output.columns = ['tbH_seas']
tbv_output.columns = ['tbV_seas']
sm_output.columns = ['sm_am_seas']

overlap_final = overlap.join(tbh_output)
overlap_final = overlap_final.join(tbv_output)
overlap_final = overlap_final.join(sm_output)

# to make sure there are no year-to-year differences due to missing data, median seasonal cycle is calculated (by day of year, per location)
overlap_joint = overlap_final.join(overlap_final.groupby(['lat', 'lon', 'doy'])[['tbH_seas', 'tbV_seas', 'sm_am_seas']].median(), on=['lat', 'lon', 'doy'], rsuffix='_med')
overlap_joint = overlap_joint.drop(['tbH_seas', 'tbV_seas', 'sm_am_seas' 't'])

# calculate the residuals for NN training
overlap_joint['dev_H'] = overlap_joint['tbh10'] - overlap_joint['tbH_seas_med']
overlap_joint['dev_V'] = overlap_joint['tbv10'] - overlap_joint['tbV_seas_med']
overlap_joint['dev_sm'] = overlap_joint['sm_am'] - overlap_joint['sm_am_seas_med']

path_to_save = '~/'
overlap_joint.to_pickle(path_to_save + 'smap_smos_overlap_3dm_seas_c_no_std.pkl', protocol = 4)

# calculate standard deviation of the residulas per location. Will be needed for uncertainty estimation NN runs
overlap_joint_std= overlap_joint.join(overlap_joint.groupby(['lat', 'lon'])[ 'dev_H', 'dev_V'].std(), on=['lat', 'lon'], rsuffix='_std_per_loc')

overlap_joint_std.to_pickle(path_to_save + 'smap_smos_overlap_3dm_seas_c.pkl', protocol = 4)

# soil moisture seasonal cycle only. by day of year
ss = overlap_joint[['day', 'lat', 'lon', 'sm_am_seas_med']]
ss = ss.drop_duplicates()
ss.to_pickle(path_to_save + 'sm_seas_cycle.pkl', protocol = 4)


print('OVERLAP: The number of points at which seasonal cycle was succesfully calculated using curve_fit: %2d' %count_c)
print('OVERLAP: The number of points at which curve_fit did not work (median used instead): %2d' %count_m)
print('OVERLAP: The number of points at which time series were too short to use curve_fit (median used instead): %2d' %count_l)


# Repeat for SMOS data

# clean up smos data and match to smap data by latitude
smos = smos[smos['tbh10']>150]
smos = smos[smos['tbv10']>150]
smos = smos[smos['lat']<80]
smos = smos[smos['lat']>-60]

smos = smos.sort_values(by = ['date', 'lat', 'lon'])


d0 = smos['date'].iloc[0]
smos['t']  = (smos['date']-d0)
smos['t']  = smos['t'].dt.days.astype('int16')


count_m = 0
count_l = 0
count_c = 0

grouped_smos = smos.groupby(['lat', 'lon'])
tbh_output = grouped_smos.apply(seas_cycle_H).reset_index()
tbv_output = grouped_smos.apply(seas_cycle_V).reset_index()

tbh_output= tbh_output.set_index('level_2')
tbv_output= tbv_output.set_index('level_2')

tbh_output = tbh_output.drop(columns=['lat', 'lon'])
tbv_output = tbv_output.drop(columns=['lat', 'lon'])

tbh_output.columns = ['tbH_seas']
tbv_output.columns = ['tbV_seas']

smos_final = smos.join(tbh_output)
smos_final = smos_final.join(tbv_output)


smos_joint = smos_final.join(smos_final.groupby(['lat', 'lon', 'doy'])[['tbH_seas', 'tbV_seas']].median(), on=['lat', 'lon', 'doy'], rsuffix='_med')
smos_joint = smos_joint.drop(['tbH_seas', 'tbV_seas', 't'])

# add soil moisture seasonal cycle
smos_joint_sc = smos_joint.merge(ss, on=['lat', 'lon', 'doy'])


smos_joint_sc['dev_H'] = smos_joint_sc['tbh10'] - smos_joint_sc['tbH_seas_med']
smos_joint_sc['dev_V'] = smos_joint_sc['tbv10'] - smos_joint_sc['tbV_seas_med']

smos_joint_sc.to_pickle(path_to_save + 'smos_3dm_seas_c_no_std.pkl', protocol = 4)

smos_joint_sc = smos_joint_sc.join(smos_joint_sc.groupby(['lat', 'lon'])[ 'dev_H', 'dev_V'].std(), on=['lat', 'lon'], rsuffix='_std_per_loc')

smos_joint_sc.to_pickle(path_to_save + 'smos_3dm_seas_c.pkl', protocol = 4)

print('SMOS: The number of points at which seasonal cycle was succesfully calculated using curve_fit: %2d' %count_c)
print('SMOS: The number of points at which curve_fit did not work (median used instead): %2d' %count_m)
print('SMOS: The number of points at which time series were too short to use curve_fit (median used instead): %2d' %count_l)
