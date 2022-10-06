# This scripts takes all outputs together and prepares for aanlysis and in-situ data comparison

# input files: data_unc_in_2010-2020.pkl, stuct_unc_in_2010-2020.pkl, data_unc_in_2002-2010.pkl, stuct_unc_in_2002-2010.pkl

# output files: data_unc_2002-2010_AUG22 - for map and for time series, for 2002-2010 and 2010-2020, structural and data uncertainty
# casm_dataset.pkl
# casm_xr.zarr
# yearly CASM_SM_' + str(year) + '.nc'

import pandas as pd
import numpy as np
import xarray as xr
import zarr

path_to_file = '~/'
path_to_save_output = '~/'

# read data uncertainty files

file_data1 =path_to_file + 'data_unc_in_2002-2010.pkl'
data_unc2002 = pd.read_pickle(file_data1)
data_unc2002['date'] = pd.to_datetime(data_unc2002['date'])
data_unc2002_short = data_unc2002[['lat', 'lon', 'date', 'data_std']]

data_unc2010 = pd.read_pickle(path_to_file + 'data_unc_in_2010-2020.pkl')
data_unc2010['date'] = pd.to_datetime(data_unc2010['date'])
data_unc2010_short = data_unc2010[['lat', 'lon', 'date', 'data_std']]

# combine two time periods, take the mean for the period where the two overlap

full_data_unc = data_unc2002_short.merge(data_unc2010_short, how = 'outer', on =['lat', 'lon', 'date'])
full_data_unc['data_uncertainty'] =full_data_unc[['data_std_x', 'data_std_y']].mean(axis = 1)
full_data_unc = full_data_unc.drop(['data_std_x', 'data_std_y'], axis=1)
full_data_unc = full_data_unc.sort_values(by='date')


# maps and time series for data

def group_from_data(data, type, columns, path_to_save_output, name_to_save):
    if type =='map':
        groupped= data.groupby(['lat', 'lon'])[columns].mean()
    else:
        groupped= data.groupby(['date'])[columns].mean()
    
    groupped = groupped.reset_index()
    groupped.to_pickle(path_to_save_output + name_to_save + '_for' + type +'.pkl', protocol = 4)
    return
    
group_from_data(data_unc2002, 'ts', ['data_mean', 'data_std', 'range_min', 'range_max'], path_to_save_output, 'data_unc_2002-2010_AUG22' )
group_from_data(data_unc2010, 'ts', ['data_mean', 'data_std', 'range_min', 'range_max'], path_to_save_output, 'data_unc_2010-2020_AUG22' )

group_from_data(data_unc2002, 'map', ['data_mean', 'data_std'], path_to_save_output, 'data_unc_2002-2010_AUG22' )
group_from_data(data_unc2010, 'map', ['data_mean', 'data_std'], path_to_save_output, 'data_unc_2010-2020_AUG22' )




# structural uncertainty data
str_unc2002 = pd.read_pickle(path_to_file + 'struct_unc_in_2002-2010.pkl')
str_unc2010 = pd.read_pickle(path_to_file + 'struct_unc_in_2010-2020.pkl')
str_unc2002 = str_unc2002[['lat', 'lon', 'date','sm_am_seas_med', 'str_mean', 'str_std']]
str_unc2010 = str_unc2010[['lat', 'lon', 'date','sm_am_seas_med', 'str_mean', 'str_std']]

# combine two time periods, take the mean for the period where the two overlap

full_str_unc = str_unc2002.merge(str_unc2010, how = 'outer', on =['lat', 'lon', 'date'])
full_str_unc['structural_uncertainty'] =full_str_unc[['str_std_x', 'str_std_y']].mean(axis = 1)
full_str_unc['CASM_soil_moisture'] =full_str_unc[['str_mean_x', 'str_mean_y']].mean(axis = 1)
full_str_unc['seasonal_cycle'] =full_str_unc[['sm_am_seas_med_x', 'sm_am_seas_med_y']].mean(axis = 1)


full_str_unc = full_str_unc.drop([ 'str_mean_x', 'str_mean_y', 'sm_am_seas_med_x', 'sm_am_seas_med_y', 'str_std_x', 'str_std_y'], axis=1)
full_str_unc = full_str_unc.sort_values(by='date')

full_dataset = full_str_unc.merge(full_data_unc, how = 'outer', on =['lat', 'lon', 'date'])


full_dataset.to_pickle(path_to_save_output + 'casm_dataset.pkl', protocol = 4)

############
# save as zarr

casm = pd.read_pickle(path_to_save_output + 'casm_dataset.pkl')
casm = casm.set_index(['date', 'lat', 'lon'])
casm_xr = xr.Dataset.from_dataframe(casm)
casm_xr.to_zarr(path_to_save_output + 'casm_xr.zarr')
#############

# save as yearly .nc with attributes

ds = xr.open_dataset(path_to_save_output + 'casm_xr.zarr')
ds['CASM_soil_moisture'].attrs = {'units':'m^3/m^3', 'long_name':'CASM soil moisture in the top soil layer'}
ds['data_uncertainty'].attrs = {'units':'m^3/m^3', 'long_name':'example aleatoric uncertainty for small (<10%) perturbation in the input data'}
ds['seasonal_cycle'].attrs = {'units':'m^3/m^3', 'long_name':'calculated auxiliary variable representing SM seasonal cycle'}
ds['structural_uncertainty'].attrs = {'units':'m^3/m^3', 'long_name':'epistemic uncertainty'}
ds.attrs = {'creation_date': 'July 2022', 'authors':'Olya Skulovich, Pierre Gentine', 'email':'os2328@columbia.edu', 'publication': 'CASM: A long-term Consistent AI-based Soil Moisture dataset based on machine learning and remote sensing', 'short_description': 'The Consistent Artificial Intelligence (AI)-based Soil Moisture (CASM) dataset is a global, consistent, and long-term, remote sensing soil moisture (SM) dataset created using machine learning. It is based on the NASA Soil Moisture Active Passive (SMAP) satellite mission SM data and is aimed at extrapolating SMAP-like quality SM back in time with previous satellite microwave platforms. CASM represents SM in the top soil layer,  it is defined on a global 25 km EASE-2 grid and for 2002-2020 with a 3-day temporal resolution'}
years = list(range(2002, 2021))

for year in years:
    da = ds.sel(date=ds.date.dt.year.isin([year]))
    print(da.info())
    filename = 'CASM_SM_' + str(year) + '.nc'
    da.to_netcdf(path_to_save_output  + filename)

################



# group by to create maps and time series:

group_from_data(full_dataset, 'ts', ['CASM_soil_moisture', 'seasonal_cycle'], path_to_save_output, 'casm_sanity_check' )
group_from_data(full_dataset, 'map', ['structural_uncertainty', 'CASM_soil_moisture'], path_to_save_output, 'casm_sanity_check' )

#######################
