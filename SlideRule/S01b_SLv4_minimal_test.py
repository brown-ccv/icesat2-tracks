
# %%
"""
This file open a ICEsat2 track applied filters and corrections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3.11
"""

# exec(open(os.environ["PYTHONSTARTUP"]).read())
# exec(open(STARTUP_2021_IceSAT2).read())
import geopandas as gpd

from sliderule import icesat2
from sliderule import earthdata
from sliderule import sliderule

import shapely

# %% Configure SL Session #

#sliderule.authenticate("brown", ps_username="mhell", ps_password="Oijaeth9quuh")

icesat2.init("slideruleearth.io", organization="brown", desired_nodes=3, time_to_live=90) #minutes


# icesat2.init("slideruleearth.io", True) 
# asset = 'nsidc-s3'

params={'srt': 1,  # Ocean classification
        'len': 25,        # 10-meter segments
        'ats':3,          # require that each segment contain photons separated by at least 5 m
        'res':10,         # return one photon every 10 m
        'track': 0,       # return all ground tracks
        'pass_invalid': True,   
        'cnf': 2,         # require classification confidence of 2 or more
        #'iterations':10,  # iterate the fit
        }

# %% single file test
track_name = '20190504201233_05580312_005_01'
ATL03_track_name = 'ATL03_'+track_name+'.h5'

gdf = icesat2.atl06p(params, resources=[ATL03_track_name])
gdf.plot()

# %% Select region and retrive batch of tracks
def create_polygons(latR, lonR):
    from shapely.geometry import Polygon, Point
    latR.sort()
    lonR.sort()
    poly_list=[{'lat':latR[ii], 'lon':lonR[jj]} for ii, jj in zip([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])]
    polygon_shapely= Polygon([(item['lon'], item['lat']) for item in poly_list])
    return {    'list':poly_list, 
                'shapely':polygon_shapely,
                'lons':lonR,
                'lats':latR}

# Southern Hemisphere test
latR, lonR = [-67.2, -62], [140.0, 155.0]

# Northern Hemisphere test
#latR, lonR = [65.0, 75.0], [140.0, 155.0]

# init polygon opbjects 
poly = create_polygons(latR, lonR)

# add domain
params['poly'] = poly['list']
params['t0'] = '2019-05-01T00:00:00'
params['t1'] = '2019-05-20T00:00:00'
#params['t1'] = '2019-06-10T00:00:00'

# get granuale list
#release = '005'
granules_list = earthdata.cmr(short_name='ATL03', version = '005', polygon=params['poly'], time_start=params['t0'], time_end=params['t1'],) 

print(granules_list)
gdf = icesat2.atl06p(params, resources=granules_list)

# %% alternative version
gdf = icesat2.atl06p(params)

# %%
