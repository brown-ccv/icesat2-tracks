import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline
#from pprint import pprint

import icepyx as ipx

# %%
downlaod_path = mconfig['paths']['scratch'] +'/SH_batch01/'

MT.mkdirs_r(downlaod_path)

# %%
date_range =['2019-06-01','2019-06-05']
region_a = ipx.Query('ATL03',[30, -70, -30, -55],date_range, \
                           start_time='09:00:00', end_time='11:59:59')

region_a.earthdata_login('mhell','mhell@ucsd.edu')
# @[49[4tK\-qBWB%5

# %%
#region_a.visualize_spatial_extent()
#region_a.order_vars.remove(all=True)

ATL03_var_list = ['dem_h', 'delta_time', 'lon_ph', 'lat_ph', 'h_ph', 'dist_ph_along' ,'dist_ph_across', 'atlas_sdp_gps_epoch', 'signal_conf_ph', 'reference_photon_index', 'segment_dist_x', 'ph_index_beg', 'ph_id_count', 'segment_id']
#ATL03_var_list = ['dem_h', 'delta_time', 'lon_ph', 'lat_ph', 'h_ph', 'dist_ph_along', 'atlas_sdp_gps_epoch', 'signal_conf_ph']

region_a.order_vars.append(var_list=ATL03_var_list)#, keyword_list=['orbit_info'])
region_a.order_vars.append( keyword_list=['orbit_info'])
region_a.order_vars.wanted

region_a.subsetparams(Coverage=region_a.order_vars.wanted)
#region_a.tracks
#region_a.file_vars

print('check how many granuals are available')
download_stars=region_a.avail_granules()
print( download_stars )
# %%

print('download '+ str(download_stars['Number of available granules']) + ' granules')
region_a.download_granules(downlaod_path)
