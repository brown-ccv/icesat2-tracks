import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
There
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')

import matplotlib.pyplot as plt
%matplotlib inline

#import m_general as M
#import m_tools as MT
import numpy as np

import m_general_ph3 as M

import netCDF4
import datetime
import os
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import h5py

import convert_GPS_time as cGPS

#import s3fs
# %%

# Python reader based on Pandas. Other reader examples available in readers.py

def save_pandas_table(table_dict, name , save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import warnings
    from pandas import HDFStore
    from pandas.io.pytables import PerformanceWarning
    warnings.filterwarnings('ignore',category=PerformanceWarning)

    with HDFStore(save_path+'/'+name+'.h5') as store:
        for name,table in table_dict.items():
                store[name]=table



def getATL03data(fileT, numpy=0, beam='gt1l', maxElev=1e6):
    # Add in a proper description of the function here

    # Open the file
    ATL03 = h5py.File(fileT, 'r')

    lons=ATL03[beam+'/heights/lon_ph'][:]
    lats=ATL03[beam+'/heights/lat_ph'][:]

    # Along track distance from equator i think.
    along_track_distance=ATL03[beam+'/heights/dist_ph_along'][:]

    #  Nathan says it's the number of seconds since the GPS epoch on midnight Jan. 6, 1980
    delta_time=ATL03[beam+'/heights/delta_time'][:]

    # #Add this value to delta time parameters to compute full gps_seconds
    atlas_epoch=ATL03['/ancillary_data/atlas_sdp_gps_epoch'][:]

    # Conversion of delta_time to a calendar date
    temp = cGPS.convert_GPS_time(atlas_epoch[0] + delta_time, OFFSET=0.0)

    # Express delta_time relative to start time of granule
    delta_time_granule=delta_time-delta_time[0]

    year = temp['year'][:].astype('int')
    month = temp['month'][:].astype('int')
    day = temp['day'][:].astype('int')
    hour = temp['hour'][:].astype('int')
    minute = temp['minute'][:].astype('int')
    second = temp['second'][:].astype('int')

    # Primary variables of interest

    # Photon height
    heights=ATL03[beam+'/heights/h_ph'][:]
    print(heights.shape)

    # Flag for signal confidence
    # column index:  0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater
    # values:
        #-- -1: Events not associated with a specific surface type
        #--  0: noise
        #--  1: buffer but algorithm classifies as background
        #--  2: low
        #--  3: medium
        #--  4: high
    signal_confidence=ATL03[beam+'/heights/signal_conf_ph'][:,2]
    print(signal_confidence.shape)

    # Add photon rate and background rate to the reader here
    ATL03.close()

    if (numpy==1):
        # list the variables you want to output here..
        return along_track_dist, elev

    else:
        dF = pd.DataFrame({'heights':heights, 'lons':lons, 'lats':lats, 'signal_confidence':signal_confidence,
                       'delta_time':delta_time_granule,'along_track_distance':along_track_distance, 'year':year, 'month':month, 'day':day, 'hour':hour, 'second':second})

        # Filter out high elevation values
        #dF = dF[(dF['signal_confidence']>2)]
        # Reset row indexing
        #dF=dF.reset_index(drop=True)
        return dF


bpath=base_path + 'data/'
path=base_path + 'data/data1/processed_ATL03_20190515060436_07170312_002_02.h5'
# %%
T =getATL03data(path, beam= 'gt2l')


Tsel= T[(T['signal_confidence']>3) & (T['heights']<100) ]


T2=Tsel[ (Tsel['delta_time']>5)  &  (Tsel['delta_time']<24) ]
# %%
M.figure_axis_xy(10, 3, view_scale=0.8)

#plt.plot(Tsel['lats'], Tsel['heights'], '.')
plt.scatter(T2['lats'], T2['heights'],0.2,  marker='.')


# %%


r_e= 6.3710E+6
dx= r_e*2*np.pi/360.0
deglon_in_m= np.cos(T2['lats']*np.pi/180.0)*dx

def derive_axis(TT):
    TT['x']=(TT['lats'].max() - TT['lats']) *dx
    TT['y']=(TT['lons'] - TT['lons'].min()) *deglon_in_m
    TT['l']=np.sqrt(TT['x']**2 + TT['y']**2)
    TT['l']= TT['l']- TT['l'].min()
    TT=TT.sort_values(by='l')
    return TT

def reduce_to_height_distance(TT, key, dx=1):

    from scipy.interpolate import interp1d

    x1 = np.arange(TT['l'].min(),TT['l'].max(), dx)
    y1 = np.interp(x1, TT['l'], TT[key] )

    return x1, y1, TT

def poly_corrent(x, y, poly_order=7, plot_flag=False):

    z = np.polyfit(x , y , poly_order)
    p = np.poly1d(z)
    if plot_flag:
        plt.plot(x,y, '.',  markersize=0.2,)
        plt.plot(x, p(x), '-',  markersize=0.2,)


    return y - p(x)


# %%
T2 = derive_axis(T2)
T2['heights_adjusted']=  poly_corrent(T2['l'], T2['heights'])
T2['heights_smth'] = T2['heights_adjusted'].rolling(100).median()
x1, y1, T3 = reduce_to_height_distance(T2,'heights_smth', dx=0.1)
point_dentisty = len(T2['l'])/ T2['l'].max() # point/ meter

x1/1e3
T2['l']
# %%
F = M.figure_axis_xy(5, 6.5, view_scale=0.8)

plt.subplot(2,1 , 1)
plt.title('Filtered Photon Heights in Sea Ice')
plt.plot(T3['l'], T3['heights_adjusted'], 'ko',  markersize=1,alpha=0.5)

plt.plot(T3['l'], T3['heights_smth'], 'bo',  markersize=1, alpha=0.5)


plt.plot(x1, y1, '-r',  markersize=2)
plt.xlim(3400, 5000)
plt.xlabel('Meters from the Sea Ice Edge')
plt.ylabel('Height Anomalie (meters)')

# %%
F = M.figure_axis_xy(4.5, 3, view_scale=0.8)

#plt.subplot(2,1, 2)
plt.plot(T3['l']/1e3, T3['heights_adjusted'], 'ko',  markersize=1,alpha=0.5, label='rar data')

#plt.plot(T3['l']/1e3, T3['heights_smth'], 'bo',  markersize=1, alpha=0.5)


plt.plot(x1/1e3, y1, '-r',  linewidth=1, label='best estimate')
plt.xlim(3400/1e3, 7000/1e3)
#plt.xlim(10000, 10500)
plt.xlabel('km from the Sea Ice Edge')
plt.ylabel('Height Anomalie (meters)')
plt.legend()
F.save_pup(path=bpath, name='filtered_photon_heights_pup')
F.save_light(path=bpath, name='filtered_photon_heights_pup_lght')
#plt.ylim(5, 12)

# %%

Gnew =xr.DataArray(y1, coords={'dist': x1}, dims='dist')
Gnew.name='filtered_photon_heights'
Gnew.to_netcdf(bpath+'/data1/filtered_photon_heights.nc')

save_pandas_table({'T':T2}, 'data1/20190515060436_07170312_002_02_filted' , bpath)
