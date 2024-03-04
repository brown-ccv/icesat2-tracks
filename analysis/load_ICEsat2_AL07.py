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


def getATL07data(fileT, numpy=0, beam='gt1r', maxElev=1e6):
    # Add in a proper description of the function here


    # Open the file
    ATL07 = h5py.File(fileT, 'r')

    lons=ATL07[beam+'/sea_ice_segments/longitude'][:]
    lats=ATL07[beam+'/sea_ice_segments/latitude'][:]

    # Along track distance from the equator crossing to the segment center.
    # I removed the first point so it's relative to the start of the beam
    along_track_distance=ATL07[beam+'/sea_ice_segments/seg_dist_x'][:] - ATL07[beam+'/sea_ice_segments/seg_dist_x'][0]
    # Height segment ID (10 km segments)
    height_segment_id=ATL07[beam+'/sea_ice_segments/height_segment_id'][:]
    #  Nathan says it's the number of seconds since the GPS epoch on midnight Jan. 6, 1980
    delta_time=ATL07[beam+'/sea_ice_segments/delta_time'][:]
    # #Add this value to delta time parameters to compute full gps_seconds
    atlas_epoch=ATL07['/ancillary_data/atlas_sdp_gps_epoch'][:]

    # Conversion of delta_time to a calendar date
    temp = cGPS.convert_GPS_time(atlas_epoch[0] + delta_time, OFFSET=0.0)

    year = temp['year'][:].astype('int')
    month = temp['month'][:].astype('int')
    day = temp['day'][:].astype('int')
    hour = temp['hour'][:].astype('int')
    minute = temp['minute'][:].astype('int')
    second = temp['second'][:].astype('int')


    # Primary variables of interest

    # Beam segment height
    elev=ATL07[beam+'/sea_ice_segments/heights/height_segment_height'][:]
    # Flag for potential leads, 0=sea ice, 1 = sea surface
    ssh_flag=ATL07[beam+'/sea_ice_segments/heights/height_segment_ssh_flag'][:]

    #Quality metrics for each segment include confidence level in the surface height estimate, which is based on the number of photons, the background noise rate, and the error measure provided by the surface-finding algorithm.
    # Height quality flag, 1 for good fit, 0 for bad
    quality=ATL07[beam+'/sea_ice_segments/heights/height_segment_quality'][:]

    elev_rms = ATL07[beam+'/sea_ice_segments/heights/height_segment_rms'][:] #RMS difference between modeled and observed photon height distribution
    seg_length = ATL07[beam+'/sea_ice_segments/heights/height_segment_length_seg'][:] # Along track length of segment
    height_confidence = ATL07[beam+'/sea_ice_segments/heights/height_segment_confidence'][:] # Height segment confidence flag
    reflectance = ATL07[beam+'/sea_ice_segments/heights/height_segment_asr_calc'][:] # Apparent surface reflectance
    ssh_flag = ATL07[beam+'/sea_ice_segments/heights/height_segment_ssh_flag'][:] # Flag for potential leads, 0=sea ice, 1 = sea surface
    seg_type = ATL07[beam+'/sea_ice_segments/heights/height_segment_type'][:] # 0 = Cloud covered
    gauss_width = ATL07[beam+'/sea_ice_segments/heights/height_segment_w_gaussian'][:] # Width of Gaussian fit


    # Geophysical corrections
    # NOTE: All of these corrections except ocean tides, DAC, and geoid undulations are applied to the ATL03 photon heights.

    # AVISO dynamic Atmospheric Correction (DAC) including inverted barometer (IB) effect (±5cm)
    dac = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_dac'][:]
    # Solid Earth Tides (±40 cm, max)
    earth = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_earth'][:]
    # Geoid (-105 to +90 m, max)
    geoid = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_geoid'][:]
    # Local displacement due to Ocean Loading (-6 to 0 cm)
    loadTide = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_load'][:]
    # Ocean Tides including diurnal and semi-diurnal (harmonic analysis),
    # and longer period tides (dynamic and self-consistent equilibrium) (±5 m)
    oceanTide = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_ocean'][:]
    # Deformation due to centrifugal effect from small variations in polar motion
    # (Solid Earth Pole Tide) (±1.5 cm, the ocean pole tide ±2mm amplitude is considered negligible)
    poleTide = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_pole'][:]
    # Mean sea surface (±2 m)
    # Taken from ICESat and CryoSat-2, see Kwok and Morison [2015])
    mss = ATL07[beam+'/sea_ice_segments/geophysical/height_segment_mss'][:]

    photon_rate = ATL07[beam+'/sea_ice_segments/stats/photon_rate'][:]

    ATL07.close()


    if (numpy==1):
        # list the variables you want to output here..
        return along_track_dist, elev

    else:
        dF = pd.DataFrame({'elev':elev, 'lons':lons, 'lats':lats, 'ssh_flag':ssh_flag,
                       'quality_flag':quality, 'delta_time':delta_time,'along_track_distance':along_track_distance, 'height_segment_id':height_segment_id, 'photon_rate':photon_rate, 'year':year, 'month':month, 'day':day, 'hour':hour, 'second':second})

        # Filter out high elevation values
        dF = dF[(dF['elev']<maxElev)]
        # Reset row indexing
        dF=dF.reset_index(drop=True)
        return dF


bpath   =   base_path + 'data/'
path    =   base_path + 'data/data1/processed_ATL07-02_20190515045104_07170301_002_01.h5'

T               = getATL07data(path, beam= 'gt2l')
T['delta_time'] = T['delta_time'] - T['delta_time'].min()
T               = T[ (T['delta_time']>5)  &  (T['delta_time']<24) ]
# %%

plt.plot(T['ssh_flag'])

r_e= 6.3710E+6
dx= r_e*2*np.pi/360.0
deglon_in_m= np.cos(T['lats']*np.pi/180.0)*dx

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
M.figure_axis_xy(5, 3, view_scale=0.8)
plt.plot(T['lats'], T['elev'], '.', markersize=1)

# %%

T2 = derive_axis(T)
T3=T2.groupby('ssh_flag')

mssh = T3['elev'].get_group(1).mean()
#T2['heights_adjusted']=  poly_corrent(T2['l'], T2['heights'])
#T2['heights_smth'] = T2['heights_adjusted'].rolling(100).median()
#x1, y1, T3 = reduce_to_height_distance(T2,'heights_smth', dx=0.1)
#point_dentisty = len(T2['l'])/ T2['l'].max() # point/ meter

# %%
F = M.figure_axis_xy(5, 6.5, view_scale=0.8)

plt.subplot(2,1 , 1)
plt.title('Freeboard')
plt.plot(T3['l'].get_group(0), T3['elev'].get_group(0)-mssh, 'ko',  markersize=1,alpha=0.5)

plt.plot(T3['l'].get_group(1), T3['elev'].get_group(1), 'g+',  markersize=1,alpha=0.5)

#plt.plot(T3['l'], T3['heights_smth'], 'bo',  markersize=1, alpha=0.5)


#plt.plot(x1, y1, '-r',  markersize=2)
#plt.xlim(150000, 9000)
plt.xlim(3400, 5000)
plt.xlabel('Meters from the Sea Ice Edge')
plt.ylabel('Height Anomalie (meters)')

plt.subplot(2,1, 2)
plt.plot(T3['l'].get_group(0), T3['elev'].get_group(0)-mssh, 'ko',  markersize=1,alpha=0.5)

plt.plot(T3['l'].get_group(1), T3['elev'].get_group(1), 'g+',  markersize=1,alpha=0.5)


#plt.plot(x1, y1, '-r',  linewidth=0.5)
#plt.xlim(3400, 7000)
#plt.xlim(10000, 10500)
plt.xlabel('Meters from the Sea Ice Edge')
plt.ylabel('Height Anomalie (meters)')

F.save_pup(path=bpath, name='freeboard_heights')
F.save_light(path=bpath, name='freeboard_heights_lght')
#plt.ylim(5, 12)

# %%

# Gnew =xr.DataArray(y1, coords={'dist': x1}, dims='dist')
# Gnew.name='freeboard_heights'
#
# Gnew.to_netcdf(bpath+'filtered_photon_heights.nc')

save_pandas_table({'T':T2}, 'freeboard_heights_filted' , bpath)


# %%
mean_freeboard=(T3['elev'].get_group(0)-mssh).rolling(1000).mean()
std_freeboard=(T3['elev'].get_group(0)-mssh).rolling(1000).std()
xx=T3['l'].get_group(0)/1e3
F = M.figure_axis_xy(5, 1.6)
plt.title('Average Free Board heigth', loc='left')
plt.fill_between(xx,  mean_freeboard-std_freeboard, mean_freeboard+std_freeboard , color='gray')
plt.plot(xx  , mean_freeboard, 'k' )
plt.xlim(10, 130)

F.save_pup(path=bpath, name='freeboard_heights_pup')
