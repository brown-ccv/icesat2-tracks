import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')
sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

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

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io

import imp
#import s3fs
# %%

# Python reader based on Pandas. Other reader examples available in readers.py


#track_name= 'ATL07-01_20190909191519_11260401_004_01'
track_name = 'processed_ATL07-02_20190515045104_07170301_002_01'
load_path   = base_path + 'data/data1/'
load_file   = load_path +track_name+'.h5'
# %

# test which beams exist:
all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
low_beams = ['gt1l',  'gt2l',  'gt3l']
high_beams = ['gt1r',  'gt2r',  'gt3r']

f         = h5py.File(load_file, 'r')
beams     = [b if b in f.keys() else None for b in all_beams]

imp.reload(io)

ATL03       =   h5py.File(load_file, 'r')

fileT=load_file
beam= high_beams[0]

imp.reload(io)
ALT07 = io.getATL07_beam(fileT)

# %%
#processed_ATL03_20190515060436_07170312_002_02
track_name = 'processed_ATL03_20190515060436_07170312_002_02'
load_path   = base_path + 'data/data1/'
load_file   = load_path +track_name+'.h5'
ATL03       =   h5py.File(load_file, 'r')

ALT03 = io.getATL03_beam(load_file)

#ALT03 = ALT03[(ALT03['signal_confidence']>1) & (ALT03['heights']<100)]
ALT03 = ALT03[(ALT03['heights']<100) & (ALT03['heights'] > -100)]

# %%
# latlims = (ALT07['ref']['latitude'].iloc[0] , ALT07['ref']['latitude'].iloc[-1] )
#
# dl = 0.08
# for ll in np.arange(latlims[0],latlims[1],dl ):
#     F = M.figure_axis_xy(7, 2, view_scale=0.8)
#
#     ALT03i = ALT03[ALT03['signal_confidence']==4 ]
#     plt.plot(ALT03i['lats'] ,ALT03i['heights'] , 'k.', markersize=0.2, alpha = 0.4, label ='ALT03 sig.level 4')
#
#     ALT03i = ALT03[(ALT03['signal_confidence']>1) & (ALT03['signal_confidence']<4)]
#     plt.plot(ALT03i['lats'] ,ALT03i['heights'] , 'g.', markersize=2, alpha = 1, label ='ALT03 sig.level 2-3')
#
#     plt.plot(ALT07['ref']['latitude'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')
#     plt.xlim(ll, ll+dl)
#     plt.legend()
#     F.save_light(path= base_path+'plots/tests/', name='ALT03_ALT07_comparison_'+ str(ll))


# %%
D = ALT07
plt.plot( D['ref']['seg_dist_x'], D['ref']['latitude'] )
D['heights']['height_segment_ssh_flag'].plot()

# %%
dist=  D['ref']['seg_dist_x']- D['ref']['seg_dist_x'][0]
segl= D['heights']['height_segment_length_seg']/10


lowest_possible_T = 2 * 2 * np.pi * np.sqrt(segl.median() / 9.81)
# %% get surface height corretions
def get_time_for_track(delta_time, atlas_epoch):
    "returns pandas dataframe"
    import pandas as pd
    import convert_GPS_time as cGPS
    # Conversion of delta_time to a calendar date
    temp = cGPS.convert_GPS_time(atlas_epoch[0] + delta_time, OFFSET=0.0)

    year = temp['year'][:].astype('int')
    month = temp['month'][:].astype('int')
    day = temp['day'][:].astype('int')
    hour = temp['hour'][:].astype('int')
    minute = temp['minute'][:].astype('int')
    second = temp['second'][:].astype('int')

    return pd.DataFrame({'year':year, 'month':month, 'day':day, 'hour':hour, 'second':second})



ALT07_cor = getATL07_height_corrections(fileT)

ALT07_cor.shape
podppd_flag.size
ALT07_cor['corrections'].keys()
# follow this correction:
# Prior to surface finding, do the following to the photon heights:
# • Remove data when padpodflag = 1. # not applied
# • Remove all TEP photons. # not applied
# • Remove the mean sea surface (MSS) heights, which are bilinearly interpolated to the
# photon locations.
# • Apply the ocean tide corrections.
# • Apply the long period equilibrium tide corrections only when there is a valid ocean tide
# correction. (i.e., when tides both are available).
# • Apply the inverted barometer (IB) corrections using met_slp from ATL09 (bilinearly
# interpolated to the photon locations) as
# hIB = -9.948(met_slp-1013.25)/1000. (meters).
# These corrections are as follows:
# h = h_ph - hMSS - hocean_tide - hlpe_tide - hIB
# (sign convention is consistent with that used in ATL03)

h_correction        =ALT07_cor['corrections']['height_segment_mss']  + ALT07_cor['corrections']['height_segment_ocean'] + ALT07_cor['corrections']['height_segment_lpe'] + ALT07_cor['corrections']['height_segment_dac']
# 'height_segment_dac' can be replaced by 'height_segment_ib', not sure ..
ALT07_cor['h_correction']  = h_correction

# %%

track_name = 'processed_ATL07-02_20190515045104_07170301_002_01'
load_path   = base_path + 'data/data1/'
load_file   = load_path +track_name+'.h5'
imp.reload(io)
ALT07 = io.getATL07_beam(load_file)


track_name = 'processed_ATL03_20190515060436_07170312_002_02'
load_path   = base_path + 'data/data1/'
load_file   = load_path +track_name+'.h5'
#ATL03       =   h5py.File(load_file, 'r')
ALT03 = io.getATL03_beam(load_file)

# %%
ALT07

ALT03 = ALT03[ (ALT03['signal_confidence']>1) & (ALT03['heights']<100) & (ALT03['heights'] > -100)]
ALT03['heights'].plot()

ALT03c =  io.getATL03_height_correction(load_file)
ALT03c['dem_h'].plot()

# cut data
ALT03c = ALT03c[ALT03c['dem_h'] < 1e5] # cute out weird references
ALT03c['dem_h'].plot()

# subtract h_ref
mask=  (ALT03c['delta_time']>= ALT03['delta_time'].min() ) & (ALT03c['delta_time'] <= ALT03['delta_time'].max())
ALT03c = ALT03c[mask]


ALT03['heights_c']= ALT03['heights'] -  np.interp( ALT03['delta_time'],ALT03c['delta_time'], ALT03c['dem_h'] )

# %%
plt.plot( ALT03c['delta_time'], ALT03c['dem_h'], zorder= 12)
plt.plot( ALT03['delta_time'], ALT03['heights'], 'k.', alpha =0.2 )

# %% test smooting


np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0)

# %%

latlims = (ALT07['ref']['latitude'].iloc[0] , ALT07['ref']['latitude'].iloc[-1] )
#latlims = (ALT07['time']['delta_time'].iloc[0] , ALT07['time']['delta_time'].iloc[-1] )
#latlims[1] -latlims[0]
dl = 1
dl = 0.07

for ll in np.arange(latlims[0],latlims[1],dl ):
    F = M.figure_axis_xy(7, 2, view_scale=0.8)

    # ALT03i = ALT03[ALT03['signal_confidence']==4 ]
    # plt.plot(ALT03i['lats'] ,ALT03i['heights'] , 'k.', markersize=0.2, alpha = 0.4, label ='ALT03 sig.level 4')

    # ALT03i = ALT03[(ALT03['signal_confidence']>1) & (ALT03['signal_confidence']<4)]
    # plt.plot(ALT03i['lats'] ,ALT03i['heights'] , 'g.', markersize=2, alpha = 1, label ='ALT03 sig.level 2-3')

    plt.plot( ALT03['lats'], ALT03['heights_c'],   'k.',  markersize= 0.3, alpha =0.2 )
    plt.plot(ALT07['ref']['latitude'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')

    # plt.plot( ALT03['delta_time'], ALT03['heights_c'],   'k.',  markersize= 0.3, alpha =0.2 )
    # plt.plot(ALT07['time']['delta_time'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')

    plt.xlim(ll, ll+dl)
    plt.ylim(-2, 2)
    #plt.legend()
    F.save_light(path= base_path+'plots/tests/', name='ALT03_ALT07_comparison_corrected'+ str(ll))
