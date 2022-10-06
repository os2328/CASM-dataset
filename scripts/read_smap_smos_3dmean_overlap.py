## This scripts combines raw SMOS TB and SMAP SM data, reads raw data, calculates 3-day mean per location, and overlaps SMAP SM and SMOS TB by location and date.

# The INPUTS are:
# 1. SMAP "raw" data

# To run this script, there must be SMAP SM data downloaded as .tif files, and regridded to 25 km EASE-2 grid
# To download SMAP as .tif, go to https://nsidc.org/data/user-resources/help-center/programmatic-data-access-guide and choose an appropriate way to download the data. You must register to be able to download.
# A simple way to download in the desired format is through https://search.earthdata.nasa.gov/search/granules?p=C2136471705-NSIDC_ECS&pg[0][v]=f&pg[0][gsk]=-start_date&q=SPL3SMP&tl=1664470448.776!3!!&long=-0.0703125
# make sure to specify .tif as the output format.

# Then regrid to EASE-2 25 km. I ised gdal from command line. Now better options are available, e.g. transform to xarray and lazily regrid it, see https://xesmf.readthedocs.io/en/latest/notebooks/Dataset.html

# 2. SMOS "raw" data
# For gridded SMOS TB go to https://www.catds.fr/sipad/. Registration is needed to download the data.

# 3. Additionally, some EASE-2 auxillary files are needed:
# 3.1. Sea - Land mask
# Available for download from here https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0609_loci_ease2/global/
# File: EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin

# 3.2. Latitudes and longitudes of EASE-2
# available at https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0772_easegrids_anc_grid_info/
# register to download
# Files: 'EASE2_M09km.lats.3856x1624x1.double' and 'EASE2_M09km.lons.3856x1624x1.double'


# This scrip outputs are smos_3dm.pkl, smap_3dm.pkl, smap_smos_overlap_3dm.pkl - all as 3-day means.


import os
from scipy.io import netcdf
import datetime
import pandas as pd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


### Sea - Land mask
# Available for download from here https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0609_loci_ease2/global/
# need to register to download
path_to_files = '~/'
fileName = path_to_files + 'EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin'
with open(fileName, mode='rb') as file: # b is important -> binary
	fileContent = file.read()
import struct
rr = struct.unpack('b' * ((len(fileContent)) ), fileContent) #return -1 - sea, 0 - land, 101 - ice
rr = np.array(rr).reshape((  584, 1388))
arr = np.flipud(rr)
land_mask = arr.ravel()


### 1. SMOS from .nc files
#
smos_path = path_to_files + '/SMOS_gridded/'
years = range(2010, 2021)
smos_dir = []
for year in years:
    year_dir = smos_path + str(year) + '/'
    smos_dir = smos_dir.append(year_dir)

SMOS_list_full = []
for d in smos_dir:
    SMOS_list = [d+f for f in os.listdir(d) if f.endswith('.nc')]
    SMOS_list_full = SMOS_list_full + SMOS_list


smos_all = pd.DataFrame({'lat': [], 'lon': [], 'bth1': [], 'bth2': [] , 'bth3': [], 'bth4': [], 'bth5': [] , 'bth6': [], 'bth7': [], 'bth8': [] , 'bth9': [], 'bth10': [], 'bth11': [] , 'bth12': [], 'bth13': [], 'bth14': [] , 'btv1': [], 'btv2': [], 'btv3': [], 'btv4': [], 'btv5': [] , 'btv6': [], 'btv7': [], 'btv8': [] , 'btv9': [], 'btv10': [], 'btv11': [] , 'btv12': [], 'btv13': [], 'btv14': [], 'date': []})


for h in SMOS_list_full:
	try:
        # read file data
        dataset = netcdf.netcdf_file(h,'r', maskandscale = True)
	except:
        print('cannot read SMOS file')
        print(h)
        continue
    tbh = dataset.variables['BT_H']
    tbh = tbh[9,:,:].copy()
    tbh1 = tbh.ravel()
	
    # masking by the fullest incidence angle (the least nan)
    # including land mask
    filt_h = (land_mask ==0) & (tbh1> 0)

    data9 = tbh1[filt_h]
    data9[data9 < 0] = np.nan
        
        # reshape latitude and longitude
    lat = dataset.variables['lat']
    lat = lat[:].copy()
    lon = dataset.variables['lon']
    lon = lon[:].copy()
    lat1 = np.broadcast_to(np.matrix(lat).transpose(), (tbh.shape[0], tbh.shape[1]))
    lon1=  np.broadcast_to(lon, (tbh.shape[0], tbh.shape[1]))

    lat1 = lat1.ravel().transpose()
    lon1 = lon1.ravel()
    latitude = lat1[filt_h]
    longitude = lon1[filt_h]
    latitude = np.asarray(latitude).reshape(-1)
    
    smos_data = pd.DataFrame({'lat': [], 'lon': [], 'bth1': [], 'bth2': [] , 'bth3': [], 'bth4': [], 'bth5': [] , 'bth6': [], 'bth7': [], 'bth8': [] , 'bth9': [], 'bth10': [], 'bth11': [] , 'bth12': [], 'bth13': [], 'bth14': [] , 'btv1': [], 'btv2': [], 'btv3': [], 'btv4': [], 'btv5': [] , 'btv6': [], 'btv7': [], 'btv8': [] , 'btv9': [], 'btv10': [], 'btv11': [] , 'btv12': [], 'btv13': [], 'btv14': [], 'date': []})

    smos_data['lat'] = latitude
    smos_data['lon'] = longitude
        
        # read rest of the angles

    for i in range(1, 15):
        tbh = dataset.variables['BT_H']
        tbh = tbh[i-1,:,:]
        tbh1 = tbh.ravel()
        data2 = tbh1[filt_h]
        data2[data2 < 0] = np.nan
        name = 'bth' +str(i)
        smos_data[name] = data2
        
        tbv = dataset.variables['BT_V']
        tbv = tbv[i-1,:,:]
        tbv1 = tbv.ravel()
        data22 = tbv1[filt_h]
        data22[data22 < 0] = np.nan
        name = 'btv' +str(i)
        smos_data[name] = data22

		
    dataset.close()
    # date from file name
    from_name_smos = h.split(path_to_files + '/SMOS_gridded/201",1)[1]
    date = from_name_smos[21:29]
    date = datetime.datetime.strptime(date, '%Y%m%d')
    smos_data['date'] = date

    smos_all = smos_all.append(smos_data)
 

    
### 2. SMAP from regridded tif

# latitude and longitude for EASE-2
# available at https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0772_easegrids_anc_grid_info/
# register to download

flat = path_to_files + 'EASE2_M09km.lats.3856x1624x1.double'
flon = path_to_files + 'EASE2_M09km.lons.3856x1624x1.double'
lat = np.fromfile(flat, dtype=np.double)
lon = np.fromfile(flon, dtype=np.double)


path = path_to_files + 'new_smap_raw/'

list_of_tifs = [f for f in os.listdir(path) if f.endswith('.tif')]

list_of_tifs.sort()

smap_All_AM = pd.DataFrame(columns=['lat', 'lon', 'date', 'sm_am'])


for ff in list_of_tifs:

        full = path+ff
        try:
            im = Image.open(full)
        except:
            print('Cannot open .tif file:')
            print(full)
            continue
            
        data1 = np.array(im)
        data = data1.ravel()
        filt_smap = (data != -9999) # filter out -9999
        data_clean = data[filt_smap]

        from_name=ff.split("_")
        date=from_name[5]
        ampm = from_name[15] # data collected at AM or PM. only AM data is used # check with the file names

        lat_cl = lat[filt_smap]
        lon_cl = lon[filt_smap]
        if ampm == 'AM':

            smap_current_day_AM = pd.DataFrame(columns=['lat', 'lon', 'date', 'sm_am'])
            smap_current_day_AM['lat'] = lat_cl
            smap_current_day_AM['lon'] = lon_cl
            smap_current_day_AM['date'] = date

            smap_current_day_AM['sm_am'] = data_clean

            smap_All_AM = smap_All_AM.append(smap_current_day_AM)
        im.close()


# 3-day mean for SMAP
smap_All_AM['date'] = pd.to_datetime(smap_All_AM['date'])
smap_All_AM = smap_All_AM.sort_values(by=['lat', 'lon', 'date'])
smap_start_date = smap_All_AM['date'].values[1]
smap_start_date = pd.to_datetime(smap_start_date, format='%Y-%m-%d')

smap_All_AM_3dm = smap_All_AM.groupby([pd.Grouper(key='date', freq='3D'), 'lat', 'lon']).mean()
smap_All_AM_3dm = smap_All_AM_3dm.reset_index()


 # apply 3-day mean for SMOS
smos_all['date'] = pd.to_datetime(smos_all['date'])
smos_all = smos_all.sort_values(by=['lat', 'lon', 'date'])
smos_3dm = smos_all.groupby([pd.Grouper(key='date', freq='3D'), 'lat', 'lon']).mean()
smos_3dm = smos_3dm.reset_index()
    

smos_start_date = smos_3dm['date'].values[1]
smos_start_date = pd.to_datetime(smos_start_date, format='%Y-%m-%d')

#it can be the case that due to beginning dates not allining, the 2 datasets will be "out of phase"
# check for that
dates_in = smos_start_date[smos_start_date['date']==smap_start_date]
if len(dates_in)<1:
    # sacrifice 1 day of smos dataset
    smos_all = smos_all[~smos_all['date']==smos_start_date]
    smos_3dm = smos_all.groupby([pd.Grouper(key='date', freq='3D'), 'lat', 'lon']).mean()
    smos_3dm = smos_3dm.reset_index()
    smos_start_date = smos_3dm['date'].values[1]
    smos_start_date = pd.to_datetime(smos_start_date, format='%Y-%m-%d')
    dates_in = smos_start_date[smos_start_date['date']==smap_start_date]
    # if it is still out of phase (there only can be 2 times - as it is 3-day mean => mod 3)
    if len(dates_in)<1:
    # sacrifice 1 more day of smos dataset
        smos_all = smos_all[~smos_all['date']==smos_start_date]
        smos_3dm = smos_all.groupby([pd.Grouper(key='date', freq='3D'), 'lat', 'lon']).mean()
        smos_3dm = smos_3dm.reset_index()
        
path_to_save = '~/'
smos_3dm.to_pickle(path_to_save + 'smos_3dm.pkl', protocol = 4)
smap_All_AM_3dm.to_pickle(path_to_save + 'smap_3dm.pkl', protocol = 4)

smap_smos_overlap = smap_All_AM_3dm.merge(smos_3dm, how = 'inner', on =['lat', 'lon', 'date'] )
smap_smos_overlap.to_pickle(path_to_save + 'smap_smos_overlap_3dm.pkl', protocol = 4)


if smap_smos_overlap.shape[0]<1:
    print('overlap went wrong. recheck the dates')
    
