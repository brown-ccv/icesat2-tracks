# %%
import os, sys

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3.11
"""
exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())

# import sys
# import logging
# import concurrent.futures
# import time
# from datetime import datetime
# import pandas as pd
import geopandas as gpd
# import matplotlib.pyplot as plt
# from pyproj import Transformer, CRS
from shapely.geometry import Polygon, Point
from sliderule import icesat2
from sliderule import sliderule
# import re
# import datetime
import numpy as np
import shapely

import matplotlib

from ipyleaflet import basemaps
from ipyleaflet import basemaps, Map, GeoData

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec
import ICEsat2_SI_tools.lanczos as lanczos
import ICEsat2_SI_tools.sliderule_converter_tools as sct
import imp
import copy
import spicke_remover
import generalized_FT as gFT

xr.set_options(display_style='text')

#matplotlib.use('agg')
# %matplotlib inline
# %matplotlib widget


plot_path = mconfig['paths']['plot'] +'sliderule_tests/'
MT.mkdirs_r(plot_path)

load_path_RGT = mconfig['paths']['analysis'] +'../analysis_db/support_files/'



# %%

# Configure Session #
icesat2.init("slideruleearth.io", True) 
asset = 'nsidc-s3'


# %% Select region and retrive batch of tracks

# create boundary based on given track information
# latR  = [np.round(ID['pars']['start']['latitude'], 1), np.round(ID['pars']['end']['latitude'], 1) ]
# lonR  = [np.round(ID['pars']['start']['longitude'], 1), np.round(ID['pars']['end']['longitude'], 1) ]
latR = [-67.2, -64.3]
lonR = [140.0, 145.0]


# latR = [65.0, 75.0]
# lonR = [140.0, 155.0]

latR.sort()
lonR.sort()

poly_test=[{'lat':latR[ii], 'lon':lonR[jj]} for ii, jj in zip([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])]
print('new ', latR, lonR)

polygon_shapely= Polygon([(item['lon'], item['lat']) for item in poly_test])


# %% load ground track data in domain:

#Gtrack = gpd.read_file(load_path_RGT + 'IS2_mission_points_NH_RGT_all.shp', mask=polygon_shapely)
Gtrack = gpd.read_file(load_path_RGT + 'IS2_mission_points_SH_RGT_all.shp', mask=poly['shapely'])

m = sct.plot_polygon(poly_test, basemap=basemaps.Esri.WorldImagery, zoom=3)
#m.add_layer(polygon_shapely, c ='red')
geo_data = GeoData(geo_dataframe = Gtrack[::5],
    #style={'color': 'red', 'radius':1},
    point_style={'radius': 0.1, 'color': 'red', 'fillOpacity': 0.1},
    name = 'rgt')
m.add_layer(geo_data)
m


# %%

Gtrack_lowest = sct.get_RGT_start_points(Gtrack)
Gtrack_highest = sct.get_RGT_end_points(Gtrack)

fig, axx = plt.subplots(1,1, figsize=(4,4))

ax = axx
ax.set_title("ATL Points")

# FIX THIS TO SHOW EACH RGT value by color
#G_lowest.plot(ax=ax, column='RGT', label='RGT', c=G_lowest['RGT'], markersize= 5, cmap='winter')
Gtrack_lowest[1:10].plot(ax=ax, label='RGT', c='red', markersize= 10)
Gtrack_highest[1:10].plot(ax=ax, label='RGT', c='black', markersize= 10)

for rgt in Gtrack['RGT'].unique()[0:10]:
    Gtrack[Gtrack['RGT'] == rgt].plot(ax=ax, c='blue', markersize= 0.8, alpha= 0.6)

#ax.legend(loc="upper left")
# Prepare coordinate lists for plotting the region of interest polygon
region_lon = [e["lon"] for e in poly_test]
region_lat = [e["lat"] for e in poly_test]
ax.plot(region_lon, region_lat, linewidth=1, color='gray');

# labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_aspect('equal')

# %%

## Generate ATL06-type segments using the ATL03-native photon classification
# Use the ocean classification for photons with a confidence parmeter to 2 or higher (low confidence or better)

params={'srt': 1,  # Ocean classification
 'len': 25,        # 10-meter segments
 'ats':3,          # require that each segment contain photons separated by at least 5 m
 'res':10,         # return one photon every 10 m
 'track': 0,       # return all ground tracks
 'pass_invalid': True,   
 'cnf': 2,         # require classification confidence of 2 or more
 #'iterations':10,  # iterate the fit
#  't0': '2019-05-02T02:12:24',  # time range (not needed in this case)
#  't1': '2019-05-02T03:00:00',
#  'poly': poly,   # polygon within which to select photons, 
}

# YAPC alternatibe
params_yapc={'srt': 1,
 'len': 20,
 'ats':3,
 'res':10,
 'track': 0,
 'pass_invalid': True,
 'cnf': -2,
 'iterations':10,}
#  't0': '2019-05-02T02:12:24',
#  't1': '2019-05-02T03:00:00',
#     #   "yapc": dict(knn=0, win_h=6, win_x=11, min_ph=4, score=100),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
#   "yapc": dict(knn=0, win_h=3, win_x=11, min_ph=4, score=50),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
#  'poly':poly}


# add domain
params['poly'] = poly_test
params['t0'] = '2019-05-01T00:00:00'
params['t1'] = '2019-05-30T00:00:00'
#params['t1'] = '2019-06-10T00:00:00'

# get granuale list
release = '005'
granules_list = icesat2.cmr(polygon=params['poly'] , time_start=params['t0'], time_end=params['t1'], version=release)

# %% download data
gdf = icesat2.atl06p(params, asset="nsidc-s3", resources=granules_list)


# %%


# make q two panel figure, on the left plot the photon postions, on the the hoffmoeller diagram
fig, axx = plt.subplots(1,3, figsize=(14,4))

# sample data
gdf2 = gdf.sample(n=12000, replace=False, random_state=1)

ax = axx[0]
ax.set_title("ATL Points")
ax.set_aspect('equal')
# FIX THIS TO SHOW EACH RGT value by color
gdf2.plot(ax=ax, column='rgt', label='RGT', c=gdf2['rgt'], markersize= 0.5, cmap='winter')
#ax.legend(loc="upper left")
# Prepare coordinate lists for plotting the region of interest polygon
region_lon = [e["lon"] for e in poly_test]
region_lat = [e["lat"] for e in poly_test]
ax.plot(region_lon, region_lat, linewidth=1, color='green');

# labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# add a red dot for each 1st point in the RGT
for rgt in gdf['rgt'].unique()[0:20]:
    tmp = gdf[gdf['rgt']==rgt]
    # 1st point
    ax.plot(tmp.iloc[0].geometry.x, tmp.iloc[0].geometry.y, 'rv', markersize= 5, label='RGT')
    # last point
    ax.plot(tmp.iloc[-1].geometry.x, tmp.iloc[-1].geometry.y, 'ks', markersize= 5, label='RGT')
    # line between first and last
    if sct.ascending_test_distance(tmp):
        axx[1].plot(tmp.geometry.x.mean(), tmp.index.mean(), 'ko', markersize= 5, label='ascending')
        ax.plot([tmp.iloc[0].geometry.x, tmp.iloc[-1].geometry.x], [tmp.iloc[0].geometry.y, tmp.iloc[-1].geometry.y], '-', color='black', linewidth=1)

    else:
        axx[1].plot( tmp.geometry.x.mean(), tmp.index.mean(), 'o', color ='orange', markersize= 5, zorder=10, label='decending')
        ax.plot([tmp.iloc[0].geometry.x, tmp.iloc[-1].geometry.x], [tmp.iloc[0].geometry.y, tmp.iloc[-1].geometry.y], '-', color='orange', linewidth=2.5)

plt.legend()

ax = axx[1]

ax.plot(gdf2.geometry.x , gdf2.index, '.', markersize=0.5)
# add labels
ax.set_xlabel('Longitude')
ax.set_ylabel('time')


ax = axx[2]
for rgt in gdf['rgt'].unique():
    tmp = gdf2[gdf2['rgt']==rgt]

    ax.plot(tmp['distance'], tmp['h_mean'],'.k', markersize=0.5)

# vertical line with min_eq_dist

plt.ylim(list(tmp['h_mean'].quantile([0.01, 0.99])))
min_eq_dist = sct.get_min_eq_dist(poly_test)
ax.axvline(min_eq_dist, color='red', linewidth=1, label= 'lower boundary')

plt.ylabel('height')
plt.xlabel('Distance from Equator')
plt.legend()
plt.show()

# %%
RGT_common = sct.check_RGT_in_domain(Gtrack_lowest, gdf)
# %%
imp.reload(sct)

for rgt in RGT_common[0:2]:
    tmp = gdf2[gdf2['rgt']==rgt]
    start_point_dist, start_point = sct.define_reference_distance_with_RGT(Gtrack_lowest, rgt, sct.ascending_test_distance(tmp))
    fig, axx = sct.plot_reference_point_coordinates(tmp, start_point_dist, start_point)


# %% main routine for defining the x coordinate and sacing table data

def make_B01_dict(table_data, split_by_beam=True):
    """
    converts a GeoDataFrame from Sliderule to GeoDataFrames for each beam witht the correct columns and names
    inputs:
        table_data: GeoDataFrame with the data
        split_by_beam: True/False. If True the data is split by beam
    returns:
        if split_by_beam:
            table_data: dict of GeoDataFrame with the data for each beam
        else:
            table_data: GeoDataFrame with the data for all beams in one table
    """
    #table_data = copy.copy(B01b_atl06_native)

    table_data.rename(columns= {
                        'n_fit_photons':'N_photos',
                        'w_surface_window_final':'signal_confidence',
                        'across':'y', 
                        #'spot':'beam',
                        }
                        , inplace=True)

    table_data['lons'] = table_data['geometry'].x
    table_data['lats'] = table_data['geometry'].y

    table_data.drop(columns=['cycle','gt','rgt', 'pflags'], inplace=True)

    if split_by_beam:
        B01b = dict()
        # this is not tested
        for spot,beam in zip([1,2,3, 4, 5, 6],['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']) : 
            ii = table_data.spot==spot
            B01b[beam] = table_data[ii]
        return B01b
    else:
        return table_data


T= dict()
for rgt in RGT_common:
    tmp = gdf[gdf['rgt']==rgt]
    # if ascending_test(tmp):
    #     print('ascending, set x to distance from first point')
    # else:
    #     print('descending, set x to distance from last point')

    # define reference point and then define 'x'
    table_data = copy.copy(tmp)

    # the reference point is defined as the most equatorward point of the polygon. 
    # It's distance from the equator is  subtracted from the distance of each photon.
    table_data = sct.define_x_coordinate_with_RGT(table_data, Gtrack_lowest)

    # fake 'across'
    table_data['across'] = table_data['x']*0 +table_data['spot']

    # add spike remover

    #Tbeam = make_B01_dict(table_data, split_by_beam=True  )
    # Tbeam['gt1r'][0:500].plot(markersize=0.5)

    # renames columns and splits beams
    T[rgt] = make_B01_dict(table_data, split_by_beam=True)

#rgt

# %% testing is re-referenceing worked
plt.figure()    

for ki,Ti in T.items():
    #Ti['gt1r'].x.plot(x='x', y='h_mean', label=ki, markersize=0.5)
    for kii,Tii in Ti.items():
        try:
            Tiis= Tii.sample(n=1000, replace=False, random_state=1)

            accending_color= 'black' if sct.ascending_test_distance(Tiis) else 'red'
            plt.plot(Tiis.geometry.x, Tiis.geometry.y, '.', markersize=0.5, label=kii, c= accending_color, alpha=0.3)
            #plt.plot(Tiis.geometry.x, Tiis['x'], '.', markersize=0.5, label=kii, c= accending_color, alpha=0.3)
            #plt.plot(Tiis['x'], Tiis.geometry.y , '.', markersize=0.5, label=kii, c= accending_color, alpha=0.3)

        except:
            #print('error with ', ki, kii)
            pass
plt.xlabel('x')
plt.ylabel('Latitude')
plt.grid()
plt.show()


# %%
