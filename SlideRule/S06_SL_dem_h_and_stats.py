# %%
"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3.11
"""
import os, sys
exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())

import geopandas as gpd

from sliderule import sliderule, icesat2, earthdata

import shapely
from ipyleaflet import basemaps, Map, GeoData

import ICEsat2_SI_tools.sliderule_converter_tools as sct
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.beam_stats as beam_stats

import spicke_remover

import h5py, imp, copy
import datetime


xr.set_options(display_style='text')

# %matplotlib inline
# %matplotlib widget


# Select region and retrive batch of tracks

track_name, batch_key, ID_flag = io.init_from_input(sys.argv) # loads standard experiment
# define file with ID:
track_name, batch_key , ID_flag = '20190219073735_08070210_005_01', 'SH_testSLsinglefile2' , False
#track_name, batch_key , ID_flag = '20190219075059_08070212_005_01', 'SH_testSLsinglefile2' , False
#track_name, batch_key , ID_flag = '20190502052058_05180312_005_01', 'SH_testSLsinglefile2' , False

#track_name, batch_key , ID_flag = '20190504201233_05580312_005_01', 'SH_testSLsinglefile2' , False


#20190502052058_05180312_005_01
plot_flag = True
hemis = batch_key.split('_')[0]
#plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name +'/'

save_path  = mconfig['paths']['work'] +'/'+batch_key+'/B01_regrid/'
MT.mkdirs_r(save_path)

save_path_json  = mconfig['paths']['work'] +'/'+ batch_key +'/A01b_ID/'
MT.mkdirs_r(save_path_json)

#ID, _, _, _ = io.init_data(track_name, batch_key, True, mconfig['paths']['work'],  )
ATL03_track_name = 'ATL03_'+track_name+'.h5'
#track_name = ID['tracks']['ATL03'][0] +'.h5'

# %% Configure SL Session #
sliderule.authenticate("brown", ps_username="mhell", ps_password="Oijaeth9quuh")
icesat2.init("slideruleearth.io", organization="brown", desired_nodes=1, time_to_live=90) #minutes


# %% plot the ground tracks in geographic location
# Generate ATL06-type segments using the ATL03-native photon classification
# Use the ocean classification for photons with a confidence parmeter to 2 or higher (low confidence or better)

params={'srt': 1,  # Ocean classification
 'len': 25,        # 10-meter segments
 'ats':3,          # require that each segment contain photons separated by at least 5 m
 'res':10,         # return one photon every 10 m
 'dist_in_seg': False, # if False units of len and res are in meters
 'track': 0,       # return all ground tracks
 'pass_invalid': False,    
 'cnt': 5,
 'sigma_r_max': 5,  # maximum standard deviation of photon in extend
 'cnf': 2,         # require classification confidence of 2 or more
 'atl03_geo_fields' : ['dem_h']
 }


# YAPC alternatibe
params_yapc={'srt': 1,
 'len': 20,
 'ats':3,
 'res':10,
 'dist_in_seg': False, # if False units of len and res are in meters 
 'track': 0,
 'pass_invalid': False,
 'cnf': 2,
 'cnt': 5,
 'maxi':10,
 'yapc': dict(knn=0, win_h=6, win_x=11, min_ph=4, score=100), # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
#   "yapc": dict(knn=0, win_h=3, win_x=11, min_ph=4, score=50),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
'atl03_geo_fields' : ['dem_h']
} 
gdf = icesat2.atl06p(params, resources=[ATL03_track_name])

# %%

imp.reload(sct)
#gdf[::100].plot(markersize=0.1, figsize=(4,6))

G = gdf.copy()#[gdf['spot'] == 1]

#G[::100].plot(markersize=0.2)

G2 = sct.correct_and_remove_height(G, 30)#.plot(markersize=0.2)
G2[::100].plot(markersize=0.2)
#Gc['h_mean'].plot(marker='.')
#G2['spot'].unique()


# %%
cdict = dict()
for s,b in zip(G2['spot'].unique(), ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']):
    cdict[s] = col.rels[b]

imp.reload(beam_stats)

#G2['h_mean_gradient'] = G2['h_mean'].diff()

font_for_pres()
F = M.figure_axis_xy(6.5, 5, view_scale=0.6)
F.fig.suptitle('title')

beam_stats.plot_ATL06_track_data(G2, cdict)


# %%
