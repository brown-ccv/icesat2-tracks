# %%
import os, sys

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3.11
"""
exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())

import sys
import logging
import concurrent.futures
import time
from datetime import datetime
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
from shapely.geometry import Polygon, Point
from sliderule import icesat2
from sliderule import sliderule
import re
import datetime
import numpy as np
import shapely
sliderule.__version__

import matplotlib

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec
import ICEsat2_SI_tools.lanczos as lanczos
import imp
import copy
import spicke_remover
import generalized_FT as gFT
xr.set_options(display_style='text')

#matplotlib.use('agg')
%matplotlib inline
%matplotlib widget


# %% define functions
def select_beam_section(data, spot, lims):
    """
    returns data from specfic beam and section between lims
    data    pd.DataFrame or GeoDataFrame
    spot    either spor number or beam key (dependinhg on dataset)
    lims    tuple of min max latitude limits
    """
    if type(data) is dict:
        mask =(data[spot].lats > lims[0]) & (data[spot].lats < lims[1]) 
        data2= data[spot][mask]
        # if abs(data2.lats[0]) > abs(data2.lats[-1]):
        #     data2.sort_values('distance', inplace=True, ascending=False)

    else:
        ii= (data.spot==spot) 
        mask = (data[ii].geometry.y > lims[0]) & (data[ii].geometry.y < lims[1]) 
        data2=  data[ii][mask]

        if abs(data2.geometry.y[0]) > abs(data2.geometry.y[-1]):
            data2.sort_values('distance', inplace=True, ascending=False)

    return data2


plot_path = mconfig['paths']['plot'] +'sliderule_tests/'
MT.mkdirs_r(plot_path)
#import s3fs
# %%
ID_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#ID_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190207002436_06190212_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190206022433_06050212_004_01', 'SH_batch02', False

#ID_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190502021224_05160312_004_01', 'SH_batch02', False

#ID_name, batch_key, test_flag =  'SH_20190208_06440212', 'SH_publish', True
#ID_name, batch_key, test_flag =  'SH_20190219_08070210', 'SH_publish', True
ID_name, batch_key, test_flag =  'SH_20190502_05160312', 'SH_publish', True

#ID_name, batch_key, test_flag =  'NH_20190311_11200203', 'NH_batch06', True
#ID_name, batch_key, test_flag =  'NH_20210312_11961005', 'NH_batch07', True

##### use init_data to load experiment metadata


ID, _, _, _ = io.init_data(ID_name, batch_key, True, mconfig['paths']['work'],  )

#track_name = 'ATL03_20190502021224_05160312_005_01'
track_name = ID['tracks']['ATL03'][0] +'.h5'
#track_name = 'ATL03_20190502021224_05160312_005_01.h5'

#print(ID_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')

all_beams   = mconfig['beams']['all_beams']

load_path_work   = mconfig['paths']['work'] +'/'+ batch_key +'/'
B2_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_regridded.h5', 'r')
B3_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_binned.h5', 'r')

B2, B3 = dict(), dict()
for b in all_beams:
    B2[b] = io.get_beam_hdf_store(B2_hdf5[b])
    B3[b] = io.get_beam_hdf_store(B3_hdf5[b])

B2_hdf5.close(), B2_hdf5.close()



# %%

# Configure Session #
#icesat2.init("icesat2sliderule.org", True)
icesat2.init("slideruleearth.io", True) #doesn't work
asset = 'nsidc-s3'

# %%
# latR=[-67.5, -64.5]
# lonR=[ 59.6,  67.7]
# print('org ', latR, lonR)

latR  = [np.round(ID['pars']['start']['latitude'], 1), np.round(ID['pars']['end']['latitude'], 1) ]
lonR  = [np.round(ID['pars']['start']['longitude'], 1), np.round(ID['pars']['end']['longitude'], 1) ]
latR.sort()
lonR.sort()
latR = [latR[0]-0.0, latR[1]+0.5]
lonR = [lonR[0]-1.0, lonR[1]+6.5]

poly=[{'lat':latR[ii], 'lon':lonR[jj]} for ii, jj in zip([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])]
print('new ', latR, lonR)

# %%

## Generate ATL06-type segments using the ATL03-native photon classification
# Use the ocean classification for photons with a confidence parmeter to 2 or higher (low confidence or better)

params={'srt': 1,  # Ocean classification
 'len': 20,        # 10-meter segments
 'ats':3,          # require that each segment contain photons separated by at least 5 m
 'res':10,         # return one photon every 5 m
 'track': 0,       # return all ground tracks
 'pass_invalid': True,   
 'cnf': 2,         # require classification confidence of 2 or more
 #'iterations':10,  # iterate the fit
 't0': '2019-05-02T02:12:24',  # time range (not needed in this case)
 't1': '2019-05-02T03:00:00',
 'poly': poly,   # polygon within which to select photons, 
}

## Generate ATL06-type segments using the YAPC photon classification

# YAPC is much more flexible than the ATL03-native classifier, but is much slower, and does not do the same sea-level-based filtering that the ATL03 ocean classsification does.

# Adjusting the YAPC 'score' parameter to something higher than 100 will return fewer photons that are more tightly clustered.

#icesat2.init("slideruleearth.io", True, organization="sliderule")
#icesat2.init("slideruleearth.io", True)#, organization="sliderule")

params_yapc={'srt': 1,
 'len': 20,
 'ats':3,
 'res':10,
 'track': 0,
 'pass_invalid': True,
 'cnf': -2,
 'iterations':10,
 't0': '2019-05-02T02:12:24',
 't1': '2019-05-02T03:00:00',
    #   "yapc": dict(knn=0, win_h=6, win_x=11, min_ph=4, score=100),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
  "yapc": dict(knn=0, win_h=3, win_x=11, min_ph=4, score=50),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
 'poly':poly}


ATL06_native =dict()
ATL06_yapc = dict()

for llen in [10, 20, 30]:
    params['len'] = llen
    params_yapc['len'] = llen

    # Run the parallel version of the ATL06 algorithm, for a specified granule:
    ATL06_native[llen] = icesat2.atl06p(params, asset="nsidc-s3", resources=[track_name])
    ATL06_yapc[llen]   = icesat2.atl06p(params_yapc, asset="nsidc-s3", resources=[track_name])#, callbacks = {"atl03rec": atl03rec_cb})

# %%
#for b in all_beams:

def init_compare_figure(B2i, B3i, lims):

    F = M.figure_axis_xy(6,4, view_scale=0.7, container=True)
    grid = F.fig.add_gridspec(2, 4, wspace=0.1, hspace=0.8)

    # Add the subplots
    axx = dict()
    axx[1] = F.fig.add_subplot(grid[0, 0:3])#, xticklabels=[])
    axx[2] = F.fig.add_subplot(grid[1, 0:3], sharex=axx[1], yticklabels=[])
    axx[3] = F.fig.add_subplot(grid[0, -1], sharey=axx[1])# yticklabels=[])
    axx[4] = F.fig.add_subplot(grid[1, -1], sharey=axx[2], yticklabels=[]) #yticklabels=[])


    # init basic data
    plt.sca(axx[1])
    plt.plot( B2i.lats, B2i.heights, '.', color = 'gray', markersize=0.8, alpha = 0.6)
    plt.plot( B3i.lats, B3i.heights, '-k', lw=0.8, label='B03 H&H 2023', zorder=12)
    #plt.plot( B3i.lats, B3i.heights, '.k', markersize = 1.2)


    plt.sca(axx[2])
    B3i_gradient = -np.gradient(B3i.heights_c_weighted_mean)
    # plt.plot( B2i.lats, B2i.heights_c, '.k', markersize=0.5)
    plt.plot( B3i.lats, B3i_gradient, '-k', lw=0.8, label='B03 H&H 2023', zorder=12)


    axx[1].set_title('Height Timeseries', loc='left')
    #ax1.set_xlabel('Latitude')
    axx[1].set_ylabel('Height')

    axx[2].set_title('Gradient', loc='left')
    axx[2].set_xlabel('Latitude')
    axx[2].set_ylabel('Gradient')

    axx[3].set_title('Scatter', loc='left')
    axx[3].set_xlabel('H&H 2023 Height')
    #axx[3].set_ylabel('Height')

    axx[4].set_title('Scatter', loc='left')
    #axx[4].set_xlabel('Gradient')
    #axx[4].set_ylabel('Gradient')

    return F, axx, B3i_gradient

# %%
def add_data(axx, data, delta_y, dx,  color, label, alpha=1.0, scatter=True):
    data_x, h_mean = data.geometry.y, data.h_mean
    #data_dx = data.dh_fit_dx  * dx
    data_dx = - np.gradient(h_mean) #* dx

    axx[1].plot( data_x, h_mean+ delta_y, color, lw=1.2, label=label, alpha=alpha)
    #axx[1].plot( data_x, h_mean+ delta_y, '.k', markersize=1.2)

    axx[2].plot( data_x, data_dx, color, lw=1.2, label=label, alpha=alpha)

    if scatter:
        if B3i.heights.size > h_mean.size:
            axx[3].scatter( B3i.heights[0:-1], h_mean, 1, c=color, label=label)
            axx[4].scatter( B3i_gradient[0:-1], data_dx, 1, c=color , label=label)
        else:
            axx[3].scatter( B3i.heights, h_mean, 1, c=color, label=label)
            axx[4].scatter( B3i_gradient, data_dx, 1, c=color , label=label)

    axx[3].plot( h_mean, h_mean, '-k', lw=0.5)
    axx[4].plot( B3i_gradient[0:-1], B3i_gradient[0:-1] , '-k', lw=0.5)


b, spot = 'gt1r', 2
B3i = B3[b][3:110]

# b, spot = 'gt3r', 6
# B3i = B3[b][2:102]

lims = [B3i.lats.min(),  B3i.lats.max() ] 
# select data given limits
B2i           = select_beam_section(B2, b, lims)

F, axx,B3i_gradient = init_compare_figure(B2i, B3i, lims)

## add more data
B3_atl06_native = select_beam_section(ATL06_native[20], spot, lims)
B3_atl06_yapc   = select_beam_section(ATL06_yapc[20], spot, lims)

delta_y = 0.0
add_data(axx, B3_atl06_native, delta_y,10,  'orange', 'ATL06 +'+str(delta_y), alpha=0.8, scatter=False) 
#axx[2].plot(B3_atl06_native.geometry.y, -np.gradient(B3_atl06_native.h_mean)+0.02, 'darkgreen' )

add_data(axx, B3_atl06_yapc, delta_y, 10,  'g', 'ATL06 YAPC +'+str(delta_y), alpha=0.8, scatter=False) 
#axx[2].plot(B3_atl06_yapc.geometry.y, -np.gradient(B3_atl06_yapc.h_mean) - 0.02 , 'lightgreen' )

axx[1].legend(ncol=3)
F.fig.suptitle('ATL06 native vs. YAPC')
plt.show()


F.save_light(name ='SlideRule_ATL06_compare_weak', path=plot_path)


# %% Test windowing ATL06_native

#b, spot = 'gt3r', 6
b, spot = 'gt1r', 2

B3i = B3[b][3:110]
lims = [B3i.lats.min(),  B3i.lats.max() ] 
# select data given limits
B2i           = select_beam_section(B2, b, lims)

F, axx, B3i_gradient = init_compare_figure(B2i, B3i, lims)

## add data
delta_y=0.0
for ll,ccol in zip([10, 20, 30], ['green', 'darkorange', 'red']):
    B3_atl06_native = select_beam_section(ATL06_native[ll], spot, lims)

    delta_y += 0.1
    add_data(axx, B3_atl06_native, delta_y,10,  ccol, 'len='+ str(ll) +'| +' +str(np.round(delta_y, 2)), alpha=0.8, scatter=False) 
    #axx[2].plot(B3_atl06_native.geometry.y, -np.gradient(B3_atl06_native.h_mean)+0.02, 'darkgreen' )

axx[1].legend(ncol=3)

axx[1].set_ylim(B3i.heights.min(), B3i.heights.max()+0.2)
axx[2].set_ylim( np.nanmin(B3i_gradient), np.nanmax(B3i_gradient))
F.fig.suptitle('ATL06 native')

F.save_light(name ='SlideRule_ATL06_native_window_weak', path=plot_path)
plt.show()

# %% Test windowing ATL06_yapc


b, spot = 'gt1r', 2

B3i = B3[b][3:102]
lims = [B3i.lats.min(),  B3i.lats.max() ] 
# select data given limits
B2i           = select_beam_section(B2, b, lims)

F, axx, B3i_gradient = init_compare_figure(B2i, B3i, lims)

## add data
delta_y=0.0
for ll,ccol in zip([10, 20, 30], ['green', 'darkorange', 'red']):
    B3_atl06_yapc = select_beam_section(ATL06_yapc[ll], spot, lims)

    delta_y += 0.1
    add_data(axx, B3_atl06_yapc, delta_y,10,  ccol, 'len='+str(ll)+'| +' +str(np.round(delta_y, 2)), alpha=0.8, scatter=False) 
    #axx[2].plot(B3_atl06_native.geometry.y, -np.gradient(B3_atl06_native.h_mean)+0.02, 'darkgreen' )

axx[1].legend(ncol=3)

axx[1].set_ylim(B3i.heights.min(), B3i.heights.max()+0.2)
axx[2].set_ylim( np.nanmin(B3i_gradient), np.nanmax(B3i_gradient))
F.fig.suptitle('ATL06 YAPC')
plt.show()
#F.save_light(name ='SlideRule_ATL06_yapc_window_weak', path=plot_path)

# %% YAPC classifications:
# --K, -k
#     Number of values for KNN algorithm
#         Use 0 for dynamic selection of neighbors
# --min-knn
#     Minimum value of K used in the KNN algorithm
# --min-ph
#     Minimum number of photons for a segment to be valid
# --min-x-spread
#     Minimum along-track spread of photon events
# --min-h-spread
#     Minimum window of heights for photon events
# --win-x
#     Along-track length of window
# --win-h
#     Height of window
#         Use 0 for dynamic window height


params_yapc={'srt': 1,
 'len': 20,
 'ats':5,
 'res':10,
 'track': 0,
 'pass_invalid': True,
 'cnf': -2,
 'iterations':10,
 't0': '2019-05-02T02:12:24',
 't1': '2019-05-02T03:00:00',
  "yapc": dict(knn=0, win_h=6, win_x=20, min_ph=4, score=100),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
 'poly':poly}


win_x_list = [10, 20, 30]
win_h_list = [0, 10, 20]

ATL06_yapc_winx = dict()
wwin_h = 10
for wwin_x in [10, 20, 30]:
        params_yapc['yapc'] = dict(knn=0,win_x=wwin_x, win_h=wwin_h, min_ph=4, score=100)
        # Run the parallel version of the ATL06 algorithm, for a specified granule:
        #case_name = str(wwin_x) +'_' +str(wwin_h)
        ATL06_yapc_winx[wwin_x]   = icesat2.atl06p(params_yapc, asset="nsidc-s3", resources=[track_name])


ATL06_yapc_winh = dict()
wwin_x = 20
for wwin_h  in [0, 10, 20]:
        params_yapc['yapc'] = dict(knn=0,win_x=wwin_x, win_h=wwin_h, min_ph=4, score=100)
        # Run the parallel version of the ATL06 algorithm, for a specified granule:
        #case_name = str(wwin_x) +'_' +str(wwin_h)
        ATL06_yapc_winh[wwin_h]   = icesat2.atl06p(params_yapc, asset="nsidc-s3", resources=[track_name])


# %%

b, spot = 'gt1r', 2
B3i = B3[b][3:102]
lims = [B3i.lats.min(),  B3i.lats.max() ] 
# select data given limits
B2i           = select_beam_section(B2, b, lims)

F, axx, B3i_gradient = init_compare_figure(B2i, B3i, lims)

## add data
delta_y=0.0
for ll,ccol in zip( list(ATL06_yapc_winx.keys()) , ['green', 'darkorange', 'red']):
    B3_atl06_yapc = select_beam_section(ATL06_yapc_winx[ll], spot, lims)

    delta_y += 0.0
    add_data(axx, B3_atl06_yapc, delta_y,10,  ccol, 'win_x='+str(ll)+'| +' +str(np.round(delta_y, 2)), alpha=0.8, scatter=False) 
    #axx[2].plot(B3_atl06_native.geometry.y, -np.gradient(B3_atl06_native.h_mean)+0.02, 'darkgreen' )

axx[1].legend(ncol=2)

axx[1].set_ylim(B3i.heights.min(), B3i.heights.max()+0.2)
axx[2].set_ylim( np.nanmin(B3i_gradient), np.nanmax(B3i_gradient))
F.fig.suptitle('ATL06 YAPC win_x comapare')
plt.show()


# win_h
F, axx, B3i_gradient = init_compare_figure(B2i, B3i, lims)

## add data
delta_y=0.0
for ll,ccol in zip( list(ATL06_yapc_winh.keys()) , ['green', 'darkorange', 'red']):
    B3_atl06_yapc = select_beam_section(ATL06_yapc_winh[ll], spot, lims)

    delta_y += 0.0
    add_data(axx, B3_atl06_yapc, delta_y,10,  ccol, 'win_h='+str(ll)+'| +' +str(np.round(delta_y, 2)), alpha=0.8, scatter=False) 
    #axx[2].plot(B3_atl06_native.geometry.y, -np.gradient(B3_atl06_native.h_mean)+0.02, 'darkgreen' )

axx[1].legend(ncol=2)

axx[1].set_ylim(B3i.heights.min(), B3i.heights.max()+0.2)
axx[2].set_ylim( np.nanmin(B3i_gradient), np.nanmax(B3i_gradient))
F.fig.suptitle('ATL06 YAPC win_h comapare')
plt.show()


# %% plot YAPC classifications for case above just for checking

# --- look up again what classifies as icesat2.SRT_OCEAN and icesat2.SRT_SEA_ICE ?
#The SRT categorization is non-exclusive. Ocean photons can be also Sea ice photons.

parms = {
    # processing parameters
    'release' :'005',
    "srt": icesat2.SRT_OCEAN,
    'time_start':'2019-05-02T02:12:24',
    'time_end':'2019-05-02T03:00:00',
    "len": 20,
    'track':1,  # select gt1l
    # classification and checks
    # still return photon segments that fail checks
    "pass_invalid": True, 
    # all photons
    "cnf": -2, 
    "yapc": dict(knn=0, win_h=10, win_x=11, min_ph=4, score=0), 
    'atl03_ph_fields' : ['dist_ph_along','dist_ph_across'],
    'poly':poly
}

gdf = icesat2.atl03s(parms, asset=asset,  resource=track_name)#granules_list)
# %%
parms = {
    # processing parameters
    'release' :'005',
    "srt": icesat2.SRT_SEA_ICE,
    'time_start':'2019-05-02T02:12:24',
    'time_end':'2019-05-02T03:00:00',
    "len": 20,
    'track':1,  # select gt1l
    # classification and checks
    # still return photon segments that fail checks
    "pass_invalid": True, 
    # all photons
    "cnf": -2, 
    "yapc": dict(knn=0, win_h=10, win_x=11, min_ph=4, score=0), 
    'atl03_ph_fields' : ['dist_ph_along','dist_ph_across'],
    'poly':poly
}

gdf_seaice = icesat2.atl03s(parms, asset=asset,  resource=track_name)#granules_list)


# %%

b, spot = 'gt1r', 2
#B3i = B3[b][600:702]
B3i = B3[b][0:102]

lims = [B3i.lats.min(),  B3i.lats.max() ] 
# select data given limits
B2i           = select_beam_section(B2, b, lims)

#F, axx, B3i_gradient = init_compare_figure(B2i, B3i, lims)

gdfi           = select_beam_section(gdf, spot, lims)
gdf_seaicei    = select_beam_section(gdf_seaice, spot, lims)

# Make 1 panel figure with B3i.heights and gdfi.height
F = M.figure_axis_xy(7.5, 3)
ax =F.ax

ax.plot( B2i.lats, B2i.heights, '.', color = 'gray', markersize=0.8, alpha = 0.6)
ax.plot( B3i.lats, B3i.heights, '-k', lw=0.8, label='B03 H&H 2023', zorder=12)

score_curoff = 10 #230
mask = gdfi['yapc_score'] > score_curoff
#ax.plot( gdfi[~mask].geometry.y , gdfi[~mask].height, '.', color='red', markersize=1.5, alpha = 0.6, label='YAPC score < '+str(score_curoff))


axx[1].plot( gdfi['segment_dist'] + gdfi['distance'], gdfi.height, '.')
hm = ax.scatter( gdfi[mask].geometry.y , gdfi[mask].height, c=gdfi[mask]['yapc_score'], s=2, cmap='viridis', vmin = 0, vmax = 256, zorder= 10)

ax.plot( gdf_seaicei.geometry.y , gdf_seaicei.height, '+', color = 'red', markersize=4, alpha = 1, zorder= 3)


plt.ylim( list(gdfi[mask].height.quantile([0.02, 0.98])) )
#plt.ylim( list(B3i.heights.quantile([0.01, 0.99])) )

plt.legend()
plt.colorbar(hm)
plt.show()

# %%
