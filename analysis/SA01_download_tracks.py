import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This is a test file, playing with the Earth data login and icepyx.

"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline
#from pprint import pprint

import icepyx as ipx
import m_tools_ph3 as  MT


# %%
downlaod_path = mconfig['paths']['scratch'] +'/SH_batch02/'
path = mconfig['paths']['analysis']+'../track_lists/'
MT.mkdirs_r(downlaod_path)

flist = MT.json_load('Batch02_alex_tracks', path)

D = pd.DataFrame(flist)

def str2dt64(s):
    return np.datetime64(s[0:4]+'-'+s[4:6]+'-'+s[6:8])

D['date'] = D[0].apply(lambda row: str2dt64(row[0:8])  )

dmin, dmax = D['date'].min(), D['date'].max()
dmin, dmax

D['RGT'] = D[1].apply(lambda row: row[0:4])
D['cycle'] = D[1].apply(lambda row: row[4:6])
D['segment'] = D[1].apply(lambda row: row[6:8])
#D['segment'].hist()

D['id'] = D[0]+'_'+D[1]
#D['id_compare'] = D[0]+'_'+
D['id_compare'] = D['RGT']+D['cycle']

Dsub = D[0:2]

# %%
# list(set(D['cyle']))
# len(D['RGT'])
# len(set(D['RGT']))

# %%

date_range =[str(dmin).split(' ')[0],str(dmax).split(' ')[0]]
region_a = ipx.Query('ATL03',[30, -70, -30, -55],date_range, \
                           start_time='00:00:00', end_time='23:59:59', \
                            tracks = list(Dsub['RGT']))

region_a.earthdata_login('mhell','mhell@ucsd.edu')
# @[49[4tK\-qBWB%5

# %%
region_a.avail_granules()
region_a.avail_granules(ids=True)

# %%
#region_a.visualize_spatial_extent()
#region_a.order_vars.remove(all=True)

ATL03_var_list = ['dem_h', 'delta_time', 'lon_ph', 'lat_ph', 'h_ph', 'dist_ph_along', 'atlas_sdp_gps_epoch', 'signal_conf_ph']
region_a.order_vars.append(var_list=ATL03_var_list)#, keyword_list=['orbit_info'])
region_a.order_vars.append( keyword_list=['orbit_info'])
#region_a.order_vars.wanted

region_a.subsetparams(Coverage=region_a.order_vars.wanted)
region_a.tracks
#region_a.file_vars

# %%
help(region_a.granules)
#region_a.granules.__dict__
#region_a.granules.avail[0].keys()

type(region_a.granules.avail)
gran_list = [i['producer_granule_id'] for i in region_a.granules.avail]

len(gran_list)

sub_set= list()
for id_wanted in Dsub['id_compare']:
    sub_set.append([i for i in gran_list if id_wanted in i][0])


gran_to_delete = list(set(gran_list) - set(sub_set))

# %%
wanted_granuals= list()
for i in region_a.granules.avail:
    if any( [id_wanted in i['producer_granule_id'] for id_wanted in Dsub['id_compare']] ):
        print(i)
        wanted_granuals.append(i)

region_a.granules.avail = wanted_granuals

#help(region_a.granules)
len(region_a.granules.avail)
region_a.avail_granules(ids=True)
region_a.granules.download(verbose= True, path=downlaod_path )


# print('check how many granuals are available')
# download_stars=region_a.avail_granules()
# print( download_stars )
# # %%
#
# print('download '+ str(download_stars['Number of available granules']) + ' granules')
# region_a.download_granules(downlaod_path)
