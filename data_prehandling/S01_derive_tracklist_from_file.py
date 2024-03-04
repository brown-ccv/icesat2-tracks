import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This script takes in a list of
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')
sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

#import m_general as M
#import m_tools as MT
import numpy as np
import os
import pandas as pd

import imp
import icepyx as ipx

# %%
path = mconfig['paths']['analysis']+'../track_lists/'

# batch   = 'Batch02_alex'
with open(path+  'alex_ATL07_filelist.txt', 'r') as f:
    contents = f.readlines()

batch   = 'batch03'
with open(path+  'batch03_ATL07_filelist.txt', 'r') as f:
    contents = f.readlines()

h5_files= list()
for l in contents:
    if '.h5' in l:
        h5_files.append(l)

file_instances = list()
for h in h5_files:
    #h.split('.')[0].split('_')
    file_instances.append(  h.split('.')[0].split('_')[1:4] )


MT.json_save(batch+'_tracks_components', path, file_instances)

#file_instances
## make dataframe and derive ID that is need to compare the data:
D = pd.DataFrame(file_instances)

def str2dt64(s):
    return np.datetime64(s[0:4]+'-'+s[4:6]+'-'+s[6:8])

D['date'] = D[0].apply(lambda row: str2dt64(row[0:8])  )

dmin, dmax = D['date'].min(), D['date'].max() # needed for icspyx modules

D['RGT'] = D[1].apply(lambda row: row[0:4])
D['cycle'] = D[1].apply(lambda row: row[4:6])
D['segment'] = D[1].apply(lambda row: row[6:8])
#D['segment'].hist()

D['id'] = D[0]+'_'+D[1]
#D['id_compare'] = D[0]+'_'+
D['id_compare'] = D['RGT']+D['cycle']

# len(D['id_compare'])
# len(set(D['id_compare']))

# %%
dx= 100
all_wanted_tracks = list()
for x in np.arange(0, int(len(D)), dx):
    Dsub = D[x:x+dx]

    print('set ', x)
    # % login to earth data  ..

    date_range =[str(dmin).split(' ')[0],str(dmax).split(' ')[0]]
    region_a = ipx.Query('ATL03',[180, -70, -180, -55],date_range, \
                               start_time='00:00:00', end_time='23:59:59', \
                                tracks = list(Dsub['RGT']))

    region_a.earthdata_login('mhell','mhell@ucsd.edu')
    # pw
    # @[49[4tK\-qBWB%5

    # % request available granuals in region and time frame
    region_a.avail_granules()
    region_a.avail_granules(ids=True)

    # % compare availabe ID's with the wanted ID's
    gran_list = [i['producer_granule_id'] for i in region_a.granules.avail]
    sub_set= list()
    for id_wanted in Dsub['id_compare']:
        sub_set.append([i for i in gran_list if id_wanted in i])

    all_possible_tracks =  [item for sublist in sub_set for item in sublist]
    print( len(all_possible_tracks), ' matching granules found')

    [all_wanted_tracks.append(i) for i in all_possible_tracks]

# %% save clean file list
MT.json_save(batch+'_ATL03_A00', path, all_wanted_tracks)
