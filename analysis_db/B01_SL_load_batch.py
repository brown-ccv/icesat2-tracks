
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

#import spicke_remover

import h5py, imp, copy

xr.set_options(display_style='text')

# %matplotlib inline
# %matplotlib widget

plot_flag = True
load_path_RGT = mconfig['paths']['analysis'] +'../analysis_db/support_files/'

batch_key = 'SH_testSL'
save_path  = mconfig['paths']['work'] +'/'+batch_key+'/B01_regrid/'
MT.mkdirs_r(save_path)

save_path_json = mconfig['paths']['work'] +'/'+ batch_key +'/A01b_ID/'

# %% Configure SL Session #

sliderule.authenticate("brown", ps_username="mhell", ps_password="Oijaeth9quuh")
icesat2.init("slideruleearth.io", organization="brown", desired_nodes=3, time_to_live=90) #minutes

# %% Select region and retrive batch of tracks

# Southern Hemisphere test
latR, lonR = [-67.2, -62], [140.0, 155.0]

# Northern Hemisphere test
#latR, lonR = [65.0, 75.0], [140.0, 155.0]

# init polygon opbjects 
poly = sct.create_polygons(latR, lonR)


# %% load ground track data in domain:
#Gtrack = gpd.read_file(load_path_RGT + 'IS2_mission_points_NH_RGT_all.shp', mask=poly['shapely'])
Gtrack = gpd.read_file(load_path_RGT + 'IS2_mission_points_SH_RGT_all.shp', mask=poly['shapely'])
Gtrack_lowest = sct.get_RGT_start_points(Gtrack)

# plot the ground tracks in geographic location 
#if plot_flag:
m = sct.plot_polygon(poly['list'], basemap=basemaps.Esri.WorldImagery, zoom=4)
#m.add_layer(poly['shapely'], c ='red')
geo_data = GeoData(geo_dataframe = Gtrack[::5],
    #style={'color': 'red', 'radius':1},
    point_style={'radius': 0.1, 'color': 'red', 'fillOpacity': 0.1},
    name = 'rgt')
m.add_layer(geo_data)
m

# %% plot the ground tracks in geographic location
# Generate ATL06-type segments using the ATL03-native photon classification
# Use the ocean classification for photons with a confidence parmeter to 2 or higher (low confidence or better)

params={'srt': 1,  # Ocean classification
 'len': 25,        # 10-meter segments
 'ats':3,          # require that each segment contain photons separated by at least 5 m
 'res':10,         # return one photon every 10 m
 'track': 0,       # return all ground tracks
 'pass_invalid': True,   
 'cnf': 2,         # require classification confidence of 2 or more
 #'iterations':10,  # iterate the fit
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
#     #   "yapc": dict(knn=0, win_h=6, win_x=11, min_ph=4, score=100),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
#   "yapc": dict(knn=0, win_h=3, win_x=11, min_ph=4, score=50),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff

# add domain
params['poly'] = poly['list']
params['t0'] = '2019-05-01T00:00:00'
params['t1'] = '2019-05-20T00:00:00'
#params['t1'] = '2019-06-10T00:00:00'

# get granuale list
granules_list = earthdata.cmr(short_name='ATL03', polygon=params['poly'], time_start=params['t0'], time_end=params['t1'],) 

# %% download data from Sliderule
gdf = icesat2.atl06p(params, resources=granules_list)

# %%
imp.reload(sct)
RGT_common = sct.check_RGT_in_domain(Gtrack_lowest, gdf)

if plot_flag:
    # sample data
    gdf_sample = gdf.sample(n=12000, replace=False, random_state=1)
    fig, axx = sct.plot_data_in_domain(gdf_sample, poly['list'])
    plt.show()

# %% main routine for defining the x coordinate and sacing table data

def make_B01_dict(table_data, split_by_beam=True, to_hdf5=False):
    """
    converts a GeoDataFrame from Sliderule to GeoDataFrames for each beam with the correct columns and names
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

    drop_columns = ['cycle','gt','rgt', 'pflags']
    if to_hdf5:
        drop_columns.append('geometry')
    table_data.drop(columns=drop_columns, inplace=True)

    if split_by_beam:
        B01b = dict()
        # this is not tested
        for spot,beam in zip([1,2,3, 4, 5, 6],['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']) : 
            ii = table_data.spot==spot
            B01b[beam] = table_data[ii]
        return B01b
    else:
        return table_data


def get_box_latitude_extend(poly):
    "return in meters the latitude extend of the polygon"
    return sct.haversine(0, abs(min(poly['lats'])), 0, abs(max(poly['lats'])) ) *1e3

def get_Nmax(poly, resolution):
    "returns maximum number of bins in the y direction"
    return np.round(get_box_latitude_extend(poly) / resolution ).astype(int)


def create_search_string(track,  cycle):
    "create search string for granule list"
    track = str(track).zfill(4)
    cycle = str(cycle).zfill(2)
    return '_' + track + cycle 


def format_density_table(D_b_size, resolution):
    "format density table"
    D_size_mean  = pd.concat(D_b_size, axis=1)/get_Nmax(poly, resolution)
    D_size_mean.loc['mean'] = D_size_mean.mean(0)
    D_size_mean.loc['no_data'] = (D_size_mean == 0).sum(0)
    D_size_mean = D_size_mean.round(2)
    D_size_mean.loc['no_data'] = D_size_mean.T['no_data'].astype(int)

    D_size_mean.name = 'Datapoint Fraction'
    return D_size_mean.T

def save_table_as_pdf(formatted_table, save_path, name):
    import pandas as pd
    import matplotlib.pyplot as plt
    # Generate a PNG image of the table using matplotlib
    F = M.figure_axis_xy(10, 5)# figsize=(10, 10))
    ax = F.ax #plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    the_table = ax.table(cellText=formatted_table.values, colLabels=formatted_table.columns, rowLabels =formatted_table.index, loc='center', fontsize=24)

    the_table.auto_set_font_size(False)
    #the_table.set_fontsize(14)

    F.fig.savefig(save_path+ '/'+ name + '.pdf')


beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
# b_size = [Ti[k].size for k in beam_list]
# b_Nmedian = [Ti[k].N_photos.median() for k in beam_list]


imp.reload(sct)

T= dict()

D_b_size = dict()
D_b_Nmedian = dict()
D_name = dict()
D_track = dict()
for rgt in RGT_common:
    tmp = gdf[gdf['rgt']==rgt]


    # define reference point and then define 'x'
    table_data = copy.copy(tmp)

    # the reference point is defined as the most equatorward point of the polygon. 
    # It's distance from the equator is  subtracted from the distance of each photon.
    table_data = sct.define_x_coordinate_with_RGT(table_data, Gtrack_lowest)

    # fake 'across'
    table_data['across'] = table_data['x']*0 +table_data['spot']

    # add spike remover

    # renames columns and splits beams
    Ti = make_B01_dict(table_data, split_by_beam=True, to_hdf5=True) 
    #T[rgt] = 
    #Ti['gt1l'].drop('geometry', axis=1, inplace=True)

    ID_name = sct.create_ID_name(tmp.iloc[0])
    print( ID_name )

    io.write_track_to_HDF5(Ti, ID_name + '_B01_binned'     , save_path) # regridding heights

    # save stats
    D_b_size[rgt] = pd.Series( [Ti[k].shape[0] for k in beam_list], name = rgt)
    D_b_Nmedian[rgt] = pd.Series( [Ti[k].N_photos.median() for k in beam_list], name = rgt)

    D_name[rgt] = ID_name
    # search for granule in granule list
    search_string = create_search_string(rgt, gdf.cycle.unique()[0])
    D_track[rgt] = [g for g in granules_list if search_string in g][0]

print('save Granule list to file')
#granules_list

MT.json_save2(name='B01_SL_batch_granule_list', path=save_path_json, data= granules_list)


print('format tables ')

D_name = pd.Series(D_name, name='ID_name')
D_track = pd.Series(D_track, name='granule')

D_size_mean = format_density_table(D_b_size, resolution=params['res'])
D_size_mean = pd.concat([D_size_mean, D_track , D_name ], axis=1)
D_size_mean.sort_values('mean', ascending=False).to_html(save_path_json + batch_key + '_point_density.html')

D_b_Nmedian = pd.concat(D_b_Nmedian, axis=1)
D_b_Nmedian.index = beam_list
D_b_Nmedian = pd.concat([D_b_Nmedian.T, D_track , D_name ], axis=1)
D_b_Nmedian.to_html(save_path_json + batch_key + '_photon_density.html')


#.to_html(save_path_json + batch_key + '_point_density_sorted.html')
print('done')

# %%
