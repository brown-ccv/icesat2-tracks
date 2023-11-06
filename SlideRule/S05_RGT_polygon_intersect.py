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

from ipyleaflet import basemaps, Map, GeoData

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec
import ICEsat2_SI_tools.lanczos as lanczos

from ICEsat2_SI_tools.read_ground_tracks import *

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


# %% Select region and retrive batch of tracks

# create boundary based on given track information
# latR  = [np.round(ID['pars']['start']['latitude'], 1), np.round(ID['pars']['end']['latitude'], 1) ]
# lonR  = [np.round(ID['pars']['start']['longitude'], 1), np.round(ID['pars']['end']['longitude'], 1) ]

ground_track_length = 20160 #km

latR = [-67.2, -64.3]
lonR = [140.0, 145.0]

latR = [25.0, 30.0]
lonR = [140.0, 150.0]

latR = [65.0, 75.0]
lonR = [140.0, 160.0]

latR.sort()
lonR.sort()

poly_test=[{'lat':latR[ii], 'lon':lonR[jj]} for ii, jj in zip([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])]
print('new ', latR, lonR)


# %% Load RGT data

polygon = make_plot_polygon(poly_test)
polygon_shapely= Polygon([(item['lon'], item['lat']) for item in poly_test])


# %% 
load_path_RGT = '/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/'
G = gpd.read_file(load_path_RGT + 'IS2_mission_points_NH_RGT_all.shp') #mask=polygon_shapely)

# %%
Gs = gpd.read_file(load_path_RGT + 'IS2_mission_points_NH_RGT_all.shp', mask=polygon_shapely)


# %% plot polygon and data
m = plot_polygon(poly_test, basemap=basemaps.Esri.WorldImagery, zoom=3)
#m.add_layer(polygon_shapely, c ='red')
geo_data = GeoData(geo_dataframe = Gs[::5],
    #style={'color': 'red', 'radius':1},
    point_style={'radius': 0.1, 'color': 'red', 'fillOpacity': 0.1},
    name = 'rgt')
m.add_layer(geo_data)
m
# %%

G_lowest = get_RGT_start_points(Gs)
G_highest = get_RGT_end_points(Gs)

# %%
fig, axx = plt.subplots(1,1, figsize=(4,4))

ax = axx
ax.set_title("ATL Points")

# FIX THIS TO SHOW EACH RGT value by color
#G_lowest.plot(ax=ax, column='RGT', label='RGT', c=G_lowest['RGT'], markersize= 5, cmap='winter')

G_lowest[0:10].plot(ax=ax, label='RGT', c='red', markersize= 10)
G_highest[0:10].plot(ax=ax, label='RGT', c='black', markersize= 10)

for rgt in Gs['RGT'].unique()[0:10]:
    Gs[Gs['RGT'] == rgt].plot(ax=ax, c='blue', markersize= 0.8, alpha= 0.6)

#ax.legend(loc="upper left")
# Prepare coordinate lists for plotting the region of interest polygon
region_lon = [e["lon"] for e in poly_test]
region_lat = [e["lat"] for e in poly_test]
ax.plot(region_lon, region_lat, linewidth=1, color='gray');

# labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_aspect('equal')
# %% Define intersect with the 65deg N/S as distance=0

# find lon/lat coodinates intersect with the 65N/S lattiude 
G
