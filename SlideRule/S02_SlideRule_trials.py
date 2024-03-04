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
#matplotlib.use('agg')
# %matplotlib inline
# %matplotlib widget


# %%

# Configure Session #
#icesat2.init("icesat2sliderule.org", True)
icesat2.init("slideruleearth.io", True) #doesn't work
asset = 'nsidc-s3'

# %%
latR=[-67.5, -64.5]
lonR=[59.6, 67.7]

poly=[{'lat':latR[ii], 'lon':lonR[jj]} for ii, jj in zip([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])]

track_name = 'ATL03_20190502021224_05160312_005_01.h5'

## Generate ATL06-type segments using the ATL03-native photon classification
# Use the ocean classification for photons with a confidence parmeter to 2 or higher (low confidence or better)

params={'srt': 1,  # Ocean classification
 'len': 10,        # 10-meter segments
 'ats':5,          # require that each segment contain photons separated by at least 5 m
 'res':5,          # return one photon every 5 m
 'track': 0,       # return all ground tracks
 'pass_invalid': True,   
 'cnf': 2,         # require classification confidence of 2 or more
 'iterations':10,  # iterate the fit
 't0': '2019-05-02T02:12:24',  # time range (not needed in this case)
 't1': '2019-05-02T03:00:00',
 'poly': poly,   # polygon within which to select photons, 
 'atl03_ph_fields' : ['dist_ph_along','dist_ph_across'],
}

# Run the parallel version of the ATL06 algorithm, for a specified granule:
gdf_sea_ice = icesat2.atl06p(params, asset="nsidc-s3", resources=[track_name])

icesat2.atl06

# %%
## Generate ATL06-type segments using the YAPC photon classification

# YAPC is much more flexible than the ATL03-native classifier, but is much slower, and does not do the same sea-level-based filtering that the ATL03 ocean classsification does.

# Adjusting the YAPC 'score' parameter to something higher than 100 will return fewer photons that are more tightly clustered.


#icesat2.init("slideruleearth.io", True, organization="sliderule")
icesat2.init("slideruleearth.io", True)#, organization="sliderule")

params={'srt': 1,
 'len': 10,
 'ats':5,
 'res':5,
 'track': 0,
 'pass_invalid': True,
 'cnf': -2,
 'iterations':10,
 't0': '2019-05-02T02:12:24',
 't1': '2019-05-02T03:00:00',
  "yapc": dict(knn=0, win_h=6, win_x=11, min_ph=4, score=100),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
'atl03_ph_fields' : ['dist_ph_along','dist_ph_across'],
 'poly':poly}

gdf_yapc = icesat2.atl06p(params, asset="nsidc-s3", resources=[track_name])#, callbacks = {"atl03rec": atl03rec_cb})

# %%  Plot the results by spot:

hf, hax=plt.subplots(2,3, sharex=True, sharey=True)
hax=hax.T.ravel()
npoints= -1
for ha, spot in zip(hax.ravel(), np.arange(1,7)):
    
    ii=(gdf_yapc.spot==spot) & (gdf_yapc.w_surface_window_final < 5)
    ha.plot( gdf_yapc.geometry.y[ii][0:npoints],     gdf_yapc.h_mean[ii][0:npoints],'.', markersize=2)

    ii=(gdf_sea_ice.spot==spot) 
    ha.plot( gdf_sea_ice.geometry.y[ii][0:npoints], gdf_sea_ice.h_mean[ii][0:npoints],'.', markersize=2)
    ha.set_title(spot)

    plt.ylim(22.5, 30)

# %% Check the ATL03 classification for a subset of the region

# Downloading a whole granule worth of ATL03 photons takes a lot of time and bandwidth, but it can be instructive to look at a small portion of one granule.  Here's how to look at the YAPC classification for one track for part of the granule



parms = {
    # processing parameters
    'release' :'005',
    "srt": icesat2.SRT_OCEAN,
    'time_start':'2019-05-02T02:12:24',
    'time_end':'2019-05-02T03:00:00',
    "len": 20,
    'track':2,  # select gt1l
    # classification and checks
    # still return photon segments that fail checks
    "pass_invalid": True, 
    # all photons
    "cnf": -2, 
    "yapc": dict(knn=0, win_h=6, win_x=11, min_ph=4, score=0), 
    'atl03_ph_fields' : ['dist_ph_along','dist_ph_across'],
    'poly':poly_sub
}


gdf = icesat2.atl03s(parms, asset=asset,  resource=track_name)#granules_list)
gdf.size
# %%
F = M.figure_axis_xy(4,3)

for ispot in range(1,5):
    ii=(gdf.spot==ispot)
    gdf2 = gdf[ii][1:200:2]

    plt.plot( gdf2['segment_dist'] + gdf2['distance'], gdf2.dist_ph_across, '.')
    #plt.plot( gdf2['distance'], gdf2.dist_ph_across, 'k.')
    plt.plot( gdf2['segment_dist'] +gdf2.dist_ph_along, gdf2.dist_ph_across+10, '.')



    #plt.plot( gdf_yapc2['distance'], gdf_yapc2['distance']*0, '.')
plt.show()

#gdf_yapc.segment_id
# %%
F = M.figure_axis_xy(4,3)

for ispot in range(1,5):


    ii=(gdf_yapc.spot==ispot)
    gdf_yapc2 = gdf_yapc[ii][1:200:2]

    #plt.plot( gdf_yapc2['distance'], gdf2.dist_ph_across, '.')
    plt.plot( gdf_yapc2['distance'], gdf_yapc2['segment_id'], '.')
    #plt.plot( gdf2.geometry.x, gdf2.geometry.y, 'k.')

plt.show()



# %%
#plt.hist(gdf.yapc_score , bins= 30)
ii=np.argsort(gdf.yapc_score)

plt.figure(); plt.scatter(gdf.geometry.y[ii], gdf.height[ii], 1, c=gdf.yapc_score[ii])
plt.colorbar()

# %%

gdf_sel = gdf[gdf.yapc_score>50]

ii=np.argsort(gdf_sel.yapc_score)
plt.figure(); plt.scatter(gdf_sel.geometry.y[ii], gdf_sel.height[ii], 1, c=gdf_sel.yapc_score[ii])
plt.colorbar()
plt.ylim(1100, 1400)

# %%

# gdf_yapc.geometry
# gdf_sea_ice.geometry
# %% plot sections
F = M.figure_axis_xy(6, 2, view_scale = 0.7)

#hf, hax=plt.subplots(2,3, sharex=True, sharey=True)
hax=[F.ax]
npoints= -1#3000
for ha, spot in zip(hax, np.arange(1,2)):
    
    ii=(gdf_yapc.spot==spot) & (gdf_yapc.w_surface_window_final < 5)
    data=  gdf_yapc.h_mean[ii][0:npoints]
    ha.plot( gdf_yapc.geometry.y[ii][0:npoints],data,'.', markersize=2)

    ii=(gdf_sea_ice.spot==spot) 
    data=  gdf_sea_ice.h_mean[ii][0:npoints]
    ha.plot( gdf_sea_ice.geometry.y[ii][0:npoints], data,'.', markersize=2)
    ha.set_title(spot)



ii=np.argsort(gdf.yapc_score)

#ha.scatter(gdf.geometry.y[ii], gdf.height[ii], 1, c=gdf.yapc_score[ii])
#plt.colorbar()
#plt.xlim(-65.1, -64.8)
#plt.ylim(data.quantile([0.01, 0.99]) )


# %%

gdf_yapc.T.index
gdf_sea_ice.T.index

gdf.T.index

gdf_sea_ice.geometry.distance()

gdf_sea_ice.geometry.y
# %%
ii=(gdf_sea_ice.spot==spot)
gdf_sea_ice[ii]['distance'].plot()
ii=(gdf_sea_ice.spot==3)
gdf_sea_ice[ii]['distance'].plot()

# %%
def pole_ward_table(T):
    """
    Returns true if table goes poleward
    hdf5_file is a an HFD5 object in read mode
    """
    if T is None:
        return None
    time = T['time']['delta_time']
    lat = T['ref']['latitude']
    print('1st lat =' + str(abs(lat.iloc[time.argmin()])) , ';last lat =' + str(abs(lat.iloc[time.argmax()])) )

    return abs(lat.iloc[time.argmax()]) > abs(lat.iloc[time.argmin()])


gdf_yapc.geometry.get_coordinates()[0:10]
gdf_yapc
# %%
gdf_sea_ice.geometry.distance(gdf_yapc.geometry)

gdf_yapc.geometry.get_coordinates()[0:10]

gdf_yapc.sample(1000).geometry.plot()
gdf2 = gdf_yapc.to_crs(3857)
#gdf3 = gdf_sea_ice.to_crs(4979)
gdf2.geometry.get_coordinates()[0:10]
gdf2.geometry.rotate
gdf2.geometry.translate
gdf2.crs
# gdf3.crs

gdf2[gdf2.spot == 1][0:30].geometry.plot(marker = '.', markersize=0.4)

gdf2.sample(1000).geometry.plot()



x0,y0 = gdf2s.geometry.x.min(), gdf2s.geometry.y.min()

dist = np.sqrt( (gdf2s.geometry.x- x0)**2  + (gdf2s.geometry.y- y0)**2 )
dist.sort_values()[0:1000].diff().plot(marker= '.')
#dist.diff()[0:100].plot(marker= '.')


# %%
gdf2 = gdf_sea_ice#.to_crs(3857)     
gdf2s = gdf2[gdf2.spot == 1]#.sample(1000)

plt.figure()

#gdf2s.sort_values('distance')['distance'].diff().hist(bins=np.arange(2.5, 20, 1))
gdf2s['distance'].diff().hist(bins=np.arange(2.5, 20, 1))

#plt.show()
# %%
