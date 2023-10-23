# %%
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline
from threadpoolctl import threadpool_info, threadpool_limits
from pprint import pprint


import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import time
import imp
import copy
import spicke_remover
import datetime
import generalized_FT as gFT
from scipy.ndimage.measurements import label

# from guppy import hpy
# h=hpy()
# h.heap()
#import s3fs
#from memory_profiler import profile
import tracemalloc

def linear_gap_fill(F, key_lead, key_int):

    """
    F pd.DataFrame
    key_lead   key in F that determined the independent coordindate
    key_int     key in F that determined the dependent data
    """
    y_g = np.array(F[key_int])

    nans, x2= np.isnan(y_g), lambda z: z.nonzero()[0]
    y_g[nans]= np.interp(x2(nans), x2(~nans), y_g[~nans])

    return y_g


# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190205231558_06030212_004_01', 'SH_batch02', False

#local track
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = 'NH_20190301_09580203', 'NH_batch05', False

#track_name, batch_key, test_flag = 'NH_20190301_09590203', 'NH_batch05', False
#track_name, batch_key, test_flag = 'SH_20190101_00630212', 'SH_batch04', False
#track_name, batch_key, test_flag = 'NH_20190301_09570205',  'NH_batch05', True
#track_name, batch_key, test_flag = 'SH_20190219_08070212',  'SH_publish', True

#track_name, batch_key, test_flag = 'NH_20190302_09830203',  'NH_batch06', True

#track_name, batch_key, test_flag = 'SH_20190219_08070210', 'SH_batchminimal', True 

#track_name, batch_key , test_flag = 'SH_20190219_08070210', 'SH_testSLsinglefile2' , True
#track_name, batch_key , test_flag = 'SH_20190502_05180312', 'SH_testSLsinglefile2' , True
#track_name, batch_key , test_flag = 'SH_20190502_05180312', 'SH_testSLsinglefile2' , True


#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

load_path   = mconfig['paths']['work'] + '/'+ batch_key +'/B01_regrid/'
load_file   = load_path + 'processed_' + ATlevel + '_' + track_name + '.h5'

save_path   = mconfig['paths']['work'] + '/'+ batch_key+ '/B02_spectra/'
save_name   = 'B02_'+track_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name 
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data

# laod with pandas
#Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #

# open with hdf5
Gd = h5py.File(load_path +'/'+track_name + '_B01_binned.h5', 'r')
#Gd.close()

# %%
imp.reload(beam_stats)

import ICEsat2_SI_tools.beam_stats as beam_stats

D  = beam_stats.derive_beam_statistics(Gd, all_beams, Lmeter=10e3, dx =10)

font_for_pres()
F = M.figure_axis_xy(6.5, 4, view_scale= 0.6  )
beam_stats.plot_beam_statistics(D, high_beams, low_beams, col.rels, track_name = track_name)
F.save_light(path = plot_path , name = 'B01b_beam_statistics.png')

# print('saved and done')



# %%
