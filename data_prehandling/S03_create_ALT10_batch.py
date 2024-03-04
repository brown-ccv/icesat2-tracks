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
#downlaod_path = mconfig['paths']['scratch'] +'/SH_batch02/'
path = mconfig['paths']['analysis']+'../track_lists/'
#MT.mkdirs_r(downlaod_path)


# batch   = 'Batch02_alex'
with open(path+  'ALT10_tracks_complete.txt', 'r') as f:
    h5_files = f.readlines()

print('total number of tracks:', len(h5_files))

all_file_names = list()
for h in h5_files:
    all_file_names.append(  h.split('/')[-1].split('.')[0] )
len(all_file_names)
#MT.json_save(batch+'_tracks_components', path, file_instances)

all_file_names_split =list()
for h in all_file_names:
    all_file_names_split.append(  h.split('_') )


#flist = MT.json_load('Batch02_alex_tracks', path)

D = pd.DataFrame(all_file_names_split, index =all_file_names , columns =['ALT', 'datestr', 'ttttccss', 'version', 'revision'] )

s = D.iloc[0]['datestr']
s
def str2dt64(s):
    return np.datetime64(s[0:4]+'-'+s[4:6]+'-'+s[6:8]+'T'+s[8:10]+':'+s[10:12]+':'+s[12:14])

D['date'] = D['datestr'].apply(lambda row: str2dt64(row)  )

dmin, dmax = D['date'].min(), D['date'].max()
dmin, dmax

D['RGT'] = D['ttttccss'].apply(lambda row: row[0:4])
D['cycle'] = D['ttttccss'].apply(lambda row: row[4:6])
D['segment'] = D['ttttccss'].apply(lambda row: int(row[6:8]))
D['hemis'] = D['ALT'].apply(lambda row: 'NH' if row[6:]=='01' else 'SH')

#D['segment'].hist()

# D['id'] = D[0]+'_'+D[1]
# #D['id_compare'] = D[0]+'_'+
# D['id_compare'] = D['RGT']+D['cycle']

D['date'].min()
D['date'].max()

# %% select wanted date range
# batch = 'batch04'
# dmin, dmax = np.datetime64('2019-01-01'), np.datetime64('2019-01-30')
# hemis = 'SH'

batch = 'batch04_test'
dmin, dmax = np.datetime64('2019-01-01'), np.datetime64('2019-01-03')
hemis = 'SH'

Dsel = D[ (D['date'] > dmin) & (D['date'] < dmax)  & (D['hemis'] == hemis)]
len(Dsel)

# Dsel = D[ (D['date'] > dmin) & (D['date'] < dmax)  & (D['hemis'] == 'NH')]
# len(Dsel)



MT.json_save(batch+'_ATL10_A00', path, list(Dsel.index))
MT.save_pandas_table(Dsel, batch+'_ATL10_A00', path)
